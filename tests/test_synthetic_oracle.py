
from src.data.oracle import (
    odor_risk,
    contamination_risk,
    physical_risk,
    relation_risks,
)


def make_item(**overrides):
    item = {
        "odor_profile": "neutral",
        "sealed": True,
        "category": "일반",
        "is_electronic": False,
        "hazard_type": "none",
        "hazard_level": 0.1,
        "leak_risk": 0.0,
        "weight_sim": 10.0,
        "max_supported_weight_sim": 100.0,
        "fragile": False,
    }
    item.update(overrides)
    return item


def test_odor_direction():
    source = make_item(odor_profile="release", sealed=False)
    target = make_item(odor_profile="absorb", sealed=False)

    assert odor_risk(source, target) == 3
    assert odor_risk(target, source) == 0


def test_odor_sealed():
    source = make_item(odor_profile="release", sealed=True)
    target = make_item(odor_profile="absorb", sealed=True)

    assert odor_risk(source, target) == 1


def test_contamination_receiver_sensitivity():
    source = make_item(
        hazard_type="other",
        hazard_level=0.7,
        leak_risk=0.5,
        sealed=False,
    )

    general = make_item(category="일반", sealed=True)
    food = make_item(category="식품", sealed=False)

    assert contamination_risk(source, food) >= contamination_risk(
        source, general
    )


def test_contamination_all_levels():
    source = make_item(
        hazard_type="toxic",
        hazard_level=0.95,
        leak_risk=0.95,
        sealed=False,
    )
    target = make_item(category="식품", sealed=False)

    assert contamination_risk(source, target) == 4


def test_physical_direction():
    heavy = make_item(weight_sim=100, max_supported_weight_sim=200)
    fragile = make_item(
        weight_sim=5,
        max_supported_weight_sim=20,
        fragile=True,
    )

    assert physical_risk(heavy, fragile) > physical_risk(
        fragile, heavy
    )


def test_relation_output():
    a = make_item()
    b = make_item()

    result = relation_risks(a, b)

    assert set(result) == {
        "odor", "contamination", "physical"
    }

    assert all(
        isinstance(value, int) and 0 <= value <= 4
        for value in result.values()
    )
