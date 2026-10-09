from services.features.video.image_to_video import find_closest_aspect_ratio_option


_LOCAL = ("21:9", "16:9", "3:2", "4:3", "1:1", "4:5", "9:16")
_API_2_3 = ("16:9", "9:16")
_API_2_5 = ("16:9", "3:2", "1:1", "9:16")


def test_find_closest_aspect_ratio_option_uses_studio_ratio_distance() -> None:
    assert find_closest_aspect_ratio_option(2560, 1080, _LOCAL) == "21:9"
    assert find_closest_aspect_ratio_option(1920, 1080, _LOCAL) == "16:9"
    assert find_closest_aspect_ratio_option(1080, 1920, _LOCAL) == "9:16"
    assert find_closest_aspect_ratio_option(1500, 1000, _LOCAL) == "3:2"
    assert find_closest_aspect_ratio_option(400, 300, _LOCAL) == "4:3"
    assert find_closest_aspect_ratio_option(32, 32, _LOCAL) == "1:1"
    assert find_closest_aspect_ratio_option(1080, 1350, _LOCAL) == "4:5"


def test_closest_aspect_stays_inside_the_cell() -> None:
    assert find_closest_aspect_ratio_option(32, 32, _API_2_3) == "9:16"
    assert find_closest_aspect_ratio_option(1080, 1350, _API_2_3) == "9:16"
    assert find_closest_aspect_ratio_option(32, 32, _API_2_5) == "1:1"
    assert find_closest_aspect_ratio_option(400, 300, _API_2_5) == "3:2"
    assert find_closest_aspect_ratio_option(1080, 1350, _API_2_5) == "1:1"
