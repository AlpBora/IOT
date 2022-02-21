import signal_capture as cap

def test_find_threshold():
    assert cap.find_threshold(cap.iq_moving) == 1000000000