from reflex.sam_format import parse_boxes

# Exemple de la doc (dev.meta.ai/docs/sam/reading-segmentation), charge utile des masques raccourcie.
DOC = ("<0f>0<|box;x1=353;y1=382;x2=453;y2=427;w=537;h=561|><|mask;x=0;y=0;data=46,101,~a|b;c=d|>"
       ",1<|box;x1=260;y1=382;x2=340;y2=427;w=537;h=561|><|mask;x=0;y=0;data=46,81,~xyz|>")


def test_parse_doc_example():
    boxes = parse_boxes(DOC)
    assert [b.obj_id for b in boxes] == [0, 1]
    assert (boxes[0].x1, boxes[0].y1, boxes[0].x2, boxes[0].y2) == (353, 382, 453, 427)
    assert boxes[0].frame_w == 537 and boxes[0].has_mask


def test_confidence_and_scaling():
    b = parse_boxes("<0f>0<|box;x1=444;y1=625;x2=679;y2=850;w=1800;h=1200;c=0.9609375|>")[0]
    assert b.confidence == 0.9609375
    s = b.scaled(900, 600)
    assert (s.x1, s.y1, s.x2, s.y2) == (222, 312, 340, 425)


def test_empty_output():
    assert parse_boxes("") == []
