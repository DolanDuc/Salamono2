import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.zones_store import Zone, ZoneStore


SQUARE = [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]


def _zone(name="Strefa", severity="DANGER", polygon=None):
    return Zone(name=name, severity=severity, polygon=polygon or SQUARE)


class TestZoneModel:
    def test_defaults(self):
        z = Zone()
        assert z.name == "Strefa"
        assert z.severity == "DANGER"
        assert z.active is True
        assert z.polygon == []
        assert len(z.id) == 8
        assert z.created_at > 0

    def test_unique_ids(self):
        assert Zone().id != Zone().id

    def test_coordinate_space_defaults_to_image(self):
        assert Zone().coordinate_space == "image"

    def test_legacy_json_without_coordinate_space_loads(self):
        # Zones persisted before the field existed must load as "image".
        z = Zone.model_validate({
            "id": "abcd1234", "name": "Stara", "severity": "DANGER",
            "polygon": SQUARE, "marker_ids": [], "active": True,
            "created_at": 1.0,
        })
        assert z.coordinate_space == "image"

    def test_world_zone_roundtrip(self, tmp_path):
        path = str(tmp_path / "z.json")
        store = ZoneStore(path)
        z = Zone(name="Wykop", coordinate_space="world",
                 polygon=[[1.0, 1.0], [4.0, 1.0], [4.0, 3.0], [1.0, 3.0]])
        store.replace("_site", [z])
        loaded = ZoneStore(path).for_camera("_site")
        assert loaded[0].coordinate_space == "world"
        assert loaded[0].polygon == [[1.0, 1.0], [4.0, 1.0], [4.0, 3.0], [1.0, 3.0]]


class TestZoneStore:
    def test_empty_for_unknown_camera(self, tmp_path):
        store = ZoneStore(str(tmp_path / "z.json"))
        assert store.for_camera("cam_x") == []
        assert store.all_cameras() == {}

    def test_replace_and_read_back(self, tmp_path):
        store = ZoneStore(str(tmp_path / "z.json"))
        z = _zone(name="Wykop A")
        saved = store.replace("cam1", [z])
        assert len(saved) == 1
        assert saved[0].name == "Wykop A"
        assert store.for_camera("cam1")[0].name == "Wykop A"

    def test_replace_overwrites(self, tmp_path):
        store = ZoneStore(str(tmp_path / "z.json"))
        store.replace("cam1", [_zone(name="Old")])
        store.replace("cam1", [_zone(name="New1"), _zone(name="New2")])
        current = store.for_camera("cam1")
        names = [z.name for z in current]
        assert names == ["New1", "New2"]

    def test_persistence_across_instances(self, tmp_path):
        path = str(tmp_path / "z.json")
        s1 = ZoneStore(path)
        s1.replace("cam1", [_zone(name="Wykop B", severity="WARNING")])
        s2 = ZoneStore(path)
        zones = s2.for_camera("cam1")
        assert len(zones) == 1
        assert zones[0].name == "Wykop B"
        assert zones[0].severity == "WARNING"

    def test_multiple_cameras_isolated(self, tmp_path):
        store = ZoneStore(str(tmp_path / "z.json"))
        store.replace("cam1", [_zone(name="A")])
        store.replace("cam2", [_zone(name="B"), _zone(name="C")])
        assert len(store.for_camera("cam1")) == 1
        assert len(store.for_camera("cam2")) == 2
        assert set(store.all_cameras().keys()) == {"cam1", "cam2"}

    def test_clear(self, tmp_path):
        store = ZoneStore(str(tmp_path / "z.json"))
        store.replace("cam1", [_zone()])
        store.clear("cam1")
        assert store.for_camera("cam1") == []
        # persisted
        store2 = ZoneStore(str(tmp_path / "z.json"))
        assert store2.for_camera("cam1") == []

    def test_returns_copies_not_refs(self, tmp_path):
        store = ZoneStore(str(tmp_path / "z.json"))
        store.replace("cam1", [_zone(name="Original")])
        zones = store.for_camera("cam1")
        zones[0].name = "Mutated"
        # store unchanged
        assert store.for_camera("cam1")[0].name == "Original"

    def test_corrupt_file_returns_empty(self, tmp_path):
        path = str(tmp_path / "z.json")
        with open(path, "w") as f:
            f.write("{not valid json")
        store = ZoneStore(path)
        assert store.all_cameras() == {}

    def test_atomic_write_via_tmp(self, tmp_path):
        path = str(tmp_path / "z.json")
        store = ZoneStore(path)
        store.replace("cam1", [_zone()])
        assert os.path.exists(path)
        # tmp file should not linger
        assert not os.path.exists(path + ".tmp")
        # file is valid JSON
        with open(path) as f:
            data = json.load(f)
        assert "cam1" in data
