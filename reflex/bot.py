"""Bot Telegram : reçoit un signalement, le passe à la chaîne locale et répond en quelques secondes.

    reflex-bot            (ou python -m reflex.bot)

Sécurité :
- seuls les comptes de TELEGRAM_ALLOWED_USERS sont servis, en conversation privée ; les autres reçoivent
  leur identifiant pour que le service informatique puisse les ajouter ;
- fichiers et textes restent en mémoire, rien n'est écrit sur le disque ;
- le journal ne contient que la décision (canal, route, score, durée), jamais le contenu ni le token.

Deux vitesses : Laya répond tout de suite ; si le cas est incertain (escalate), l'agent Meta (reflex/agent.py)
l'examine en arrière-plan et le bot envoie un second message avec son verdict et l'image annotée.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Callable, Optional

from telegram import LinkPreviewOptions, Message, Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from .config import Settings, get_settings
from .pipeline import Analysis, Pipeline
from .replies import format_finding, format_reply, texts
from .triage import ESCALATE

log = logging.getLogger("reflex.bot")

MAX_FILE_MB = 10
AGENT_SLOTS = 2  # examens simultanés au plus (navigateur + appels Meta)
# Les liens des réponses sont désarmés ; on coupe en plus l'aperçu, pour que Telegram n'aille jamais sur la page.
NO_PREVIEW = LinkPreviewOptions(is_disabled=True)


def is_allowed(settings: Settings, update: Update) -> bool:
    user = update.effective_user
    return user is not None and user.id in settings.telegram_allowed_users


async def _guard(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    settings: Settings = context.bot_data["settings"]
    if is_allowed(settings, update):
        return True
    uid = update.effective_user.id if update.effective_user else "?"
    log.warning("Accès refusé à l'identifiant Telegram %s", uid)
    if update.effective_message:
        await update.effective_message.reply_text(texts(settings.bot_lang)["denied"].format(uid=uid))
    return False


async def _download(context: ContextTypes.DEFAULT_TYPE, msg: Message, file_id: str,
                    size: Optional[int]) -> Optional[bytes]:
    if size and size > MAX_FILE_MB * 1024 * 1024:
        await msg.reply_text(texts(context.bot_data["settings"].bot_lang)["too_big"].format(mb=MAX_FILE_MB))
        return None
    file = await context.bot.get_file(file_id)
    return bytes(await file.download_as_bytearray())


async def _investigate(msg: Message, context: ContextTypes.DEFAULT_TYPE, analysis: Analysis,
                      image: Optional[bytes]) -> None:
    """Second message : le verdict de l'agent Meta, avec l'image annotée quand il y en a une."""
    settings: Settings = context.bot_data["settings"]
    async with context.bot_data["agent_slots"]:
        await context.bot.send_chat_action(msg.chat_id, ChatAction.TYPING)
        try:
            finding = await asyncio.to_thread(context.bot_data["agent"].investigate, analysis, image, settings.bot_lang)
        except Exception:
            log.exception("Échec de l'agent")
            await msg.reply_text(texts(settings.bot_lang)["agent_failed"])
            return
    page = "-" if finding.visited is None else ("ok" if finding.visited.ok else "échec")
    log.info("agent verdict=%s garde-fou=%s page=%s sam=%d %.0f ms %.4f $", finding.verdict, finding.guardrail,
             page, len(finding.highlighted), finding.latency_ms, finding.cost_usd)
    caption = format_finding(finding, settings.bot_lang)
    if finding.image:
        await msg.reply_photo(photo=finding.image, caption=caption)
    else:
        await msg.reply_text(caption, link_preview_options=NO_PREVIEW)


async def _analyze_and_reply(update: Update, context: ContextTypes.DEFAULT_TYPE,
                             work: Callable[[], Analysis], error_key: str = "error",
                             image: Optional[bytes] = None) -> None:
    settings: Settings = context.bot_data["settings"]
    msg = update.effective_message
    await context.bot.send_chat_action(msg.chat_id, ChatAction.TYPING)
    try:
        # Laya et l'OCR bloquent : on les fait tourner hors de la boucle asyncio.
        analysis = await asyncio.to_thread(work)
    except Exception:
        log.exception("Échec de l'analyse")
        await msg.reply_text(texts(settings.bot_lang)[error_key])
        return
    log.info("signalement canal=%s route=%s p=%s %.0f ms", analysis.channel, analysis.route,
             "-" if analysis.p_phishing is None else f"{analysis.p_phishing:.3f}", analysis.latency_ms)
    investigate = context.bot_data.get("agent") is not None and analysis.route == ESCALATE
    await msg.reply_text(format_reply(analysis, settings.bot_lang, investigating=investigate),
                         link_preview_options=NO_PREVIEW)
    if investigate:
        # En arrière-plan : le bot continue de répondre aux autres pendant l'examen.
        context.application.create_task(_investigate(msg, context, analysis, image), update=update)


# --------------------------------------------------------------------------- handlers
async def on_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await _guard(update, context):
        await update.effective_message.reply_text(texts(context.bot_data["settings"].bot_lang)["start"])


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _guard(update, context):
        return
    pipeline: Pipeline = context.bot_data["pipeline"]
    text = update.effective_message.text
    await _analyze_and_reply(update, context, lambda: pipeline.from_text(text))


async def on_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _guard(update, context):
        return
    msg = update.effective_message
    photo = msg.photo[-1]  # la plus grande taille envoyée par Telegram
    data = await _download(context, msg, photo.file_id, photo.file_size)
    if data is not None:
        pipeline: Pipeline = context.bot_data["pipeline"]
        await _analyze_and_reply(update, context, lambda: pipeline.from_image(data), image=data)


async def on_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Image envoyée « comme fichier » (sans compression, meilleure pour l'OCR) ou e-mail .eml."""
    if not await _guard(update, context):
        return
    msg = update.effective_message
    doc = msg.document
    mime, name = (doc.mime_type or "").lower(), (doc.file_name or "").lower()
    pipeline: Pipeline = context.bot_data["pipeline"]
    if mime.startswith("image/"):
        work = pipeline.from_image
    elif name.endswith(".eml") or mime == "message/rfc822":
        work = pipeline.from_email
    else:
        await msg.reply_text(texts(context.bot_data["settings"].bot_lang)["unsupported"])
        return
    data = await _download(context, msg, doc.file_id, doc.file_size)
    if data is not None:
        await _analyze_and_reply(update, context, lambda: work(data),
                                 image=data if work == pipeline.from_image else None)


async def on_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _guard(update, context):
        return
    msg = update.effective_message
    audio = msg.voice or msg.audio
    data = await _download(context, msg, audio.file_id, audio.file_size)
    if data is not None:
        pipeline: Pipeline = context.bot_data["pipeline"]
        await _analyze_and_reply(update, context, lambda: pipeline.from_voice(data), error_key="voice_failed")


async def on_other(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await _guard(update, context):
        await update.effective_message.reply_text(texts(context.bot_data["settings"].bot_lang)["unsupported"])


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    log.error("Erreur du bot", exc_info=context.error)


# --------------------------------------------------------------------------- application
def build_app(settings: Settings, pipeline: Pipeline, agent=None) -> Application:
    app = Application.builder().token(settings.telegram_token).build()
    app.bot_data.update(settings=settings, pipeline=pipeline, agent=agent, agent_slots=asyncio.Semaphore(AGENT_SLOTS))
    private = filters.ChatType.PRIVATE  # dans un groupe, le bot ne répond pas
    app.add_handler(CommandHandler(["start", "help"], on_start, filters=private))
    app.add_handler(MessageHandler(private & filters.PHOTO, on_photo))
    app.add_handler(MessageHandler(private & filters.Document.ALL, on_document))
    app.add_handler(MessageHandler(private & (filters.VOICE | filters.AUDIO), on_voice))
    app.add_handler(MessageHandler(private & filters.TEXT & ~filters.COMMAND, on_text))
    app.add_handler(MessageHandler(private, on_other))
    app.add_error_handler(on_error)
    return app


def main() -> None:
    logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.INFO)
    # httpx journalise chaque requête avec son URL, qui contient le token du bot (httpx2 : Hugging Face, Meta).
    for name in ("httpx", "httpx2"):
        logging.getLogger(name).setLevel(logging.WARNING)
    settings = get_settings()
    if not settings.telegram_token:
        raise SystemExit("TELEGRAM_BOT_TOKEN manquant dans .env")
    if not settings.telegram_allowed_users:
        log.warning("TELEGRAM_ALLOWED_USERS est vide : le bot refuse tout le monde et répond avec l'identifiant à ajouter.")
    pipeline = Pipeline(settings)
    log.info("Chargement de Laya (%s)…", settings.laya_model)
    pipeline.warm_up()
    agent = None
    if settings.agent_enabled and settings.meta_api_key:
        from .agent import Investigator

        agent = Investigator(settings)
        log.info("Agent Meta actif sur les cas incertains (%s, %s).", settings.spark_model, settings.sam_model)
    else:
        log.warning("Agent Meta inactif (AGENT_ENABLED=0 ou MODEL_API_KEY vide) : les cas incertains restent sans examen.")
    log.info("Bot prêt. Ctrl+C pour l'arrêter.")
    build_app(settings, pipeline, agent).run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
