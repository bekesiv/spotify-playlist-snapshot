"""Unit tests for spotify_playlist_snapshot.snapshot module."""
import logging
import sqlite3
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from spotify_playlist_snapshot.snapshot import (
    CONFIG_FILENAME,
    DEFAULT_REDIRECT_URI,
    DEFAULT_CONCURRENCY,
    DEFAULT_RETRIES,
    SnapshotDb,
    ChangeLogger,
    SpotifyFetcher,
    get_configuration,
    get_arguments,
    main,
)


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

FAKE_PLAYLIST_RESPONSE_PAGE1 = {
    "items": [
        {"id": "pl_1", "name": "Chill Vibes"},
        {"id": "pl_2", "name": "Workout Mix"},
    ],
    "total": 2,
}

FAKE_TRACK_RESPONSE = {
    "items": [
        {
            "added_at": "2024-06-15T10:30:00Z",
            "track": {
                "id": "tr_1",
                "name": "Song One",
                "disc_number": 1,
                "track_number": 3,
                "is_local": False,
                "album": {"id": "al_1", "name": "Album One"},
                "artists": [
                    {"id": "ar_1", "name": "Artist A"},
                    {"id": "ar_2", "name": "Artist B"},
                ],
            },
        },
        {
            "added_at": "2024-07-01T08:00:00Z",
            "track": {
                "id": "tr_2",
                "name": "Song Two",
                "disc_number": 1,
                "track_number": 1,
                "is_local": False,
                "album": {"id": "al_2", "name": "Album Two"},
                "artists": [{"id": "ar_3", "name": "Artist C"}],
            },
        },
    ]
}

FAKE_TRACK_RESPONSE_WITH_NULL = {
    "items": [
        {
            "added_at": "2024-06-15T10:30:00Z",
            "track": {"id": None, "name": "Unavailable"},
        },
        {
            "added_at": "2024-08-01T00:00:00Z",
            "track": {
                "id": "tr_3",
                "name": "Valid Song",
                "disc_number": 1,
                "track_number": 5,
                "is_local": True,
                "album": {"id": "al_3", "name": "Album Three"},
                "artists": [{"id": "ar_4", "name": "Artist D"}],
            },
        },
    ]
}

FAKE_TRACK_RESPONSE_WITH_DUPLICATE = {
    "items": [
        FAKE_TRACK_RESPONSE["items"][0],
        FAKE_TRACK_RESPONSE["items"][1],
        {
            "added_at": "2024-09-01T12:00:00Z",
            "track": FAKE_TRACK_RESPONSE["items"][0]["track"],
        },
    ]
}


def _make_track_dict(**overrides) -> dict:
    """Helper to build a track dict with sensible defaults."""
    base = {
        "playlist_id": "pl_1",
        "playlist_name": "Test Playlist",
        "added_at": "2024-06-15_10:30:00",
        "track_id": "tr_1",
        "title": "Song One",
        "disc_number": "1",
        "track_number": "3",
        "is_local": "False",
        "album_id": "al_1",
        "album_title": "Album One",
        "artist_id": "ar_1",
        "artist_name": "Artist A",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_spotify():
    """Create a SpotifyFetcher instance with mocked internals."""
    with patch("spotify_playlist_snapshot.snapshot.SpotifyOAuth"):
        sp = SpotifyFetcher.__new__(SpotifyFetcher)
        sp.playlistmap = {}
        sp.concurrency = DEFAULT_CONCURRENCY
        sp.retries = DEFAULT_RETRIES
        sp.auth_manager = MagicMock()
        sp.auth_manager.get_access_token.return_value = "fake_token"
        sp._session = MagicMock()
        return sp


@pytest.fixture
def db(tmp_path):
    """Provide a fresh SnapshotDb backed by a temp directory."""
    with SnapshotDb(str(tmp_path / "test.db")) as database:
        yield database


@pytest.fixture(autouse=True)
def _clean_logger():
    """Remove all handlers from the snapshot logger between tests."""
    logger = logging.getLogger('spotify_snapshot')
    yield
    logger.handlers.clear()


# ---------------------------------------------------------------------------
# _group_by_track_id
# ---------------------------------------------------------------------------

class TestGroupByTrackId:
    def test_groups_unique(self):
        tracks = [_make_track_dict(track_id="a"), _make_track_dict(track_id="b")]
        groups = ChangeLogger._group_by_track_id(tracks)
        assert set(groups.keys()) == {"a", "b"}
        assert len(groups["a"]) == 1

    def test_groups_duplicates(self):
        tracks = [
            _make_track_dict(track_id="a", added_at="t1"),
            _make_track_dict(track_id="a", added_at="t2"),
            _make_track_dict(track_id="b"),
        ]
        groups = ChangeLogger._group_by_track_id(tracks)
        assert len(groups["a"]) == 2
        assert len(groups["b"]) == 1

    def test_empty(self):
        assert ChangeLogger._group_by_track_id([]) == {}

    def test_accepts_iterable(self):
        gen = (d for d in [_make_track_dict(track_id="x")])
        groups = ChangeLogger._group_by_track_id(gen)
        assert len(groups["x"]) == 1


# ---------------------------------------------------------------------------
# SnapshotDb
# ---------------------------------------------------------------------------

class TestSnapshotDb:
    def test_creates_tracks_table(self, tmp_path):
        with SnapshotDb(str(tmp_path / "new.db")) as database:
            cursor = database.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='tracks'"
            )
            assert cursor.fetchone() is not None

    def test_idempotent(self, tmp_path):
        db_path = str(tmp_path / "new.db")
        with SnapshotDb(db_path):
            pass
        with SnapshotDb(db_path) as database:
            cursor = database.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='tracks'"
            )
            assert cursor.fetchone() is not None

    def test_context_manager_closes(self, tmp_path):
        db_path = str(tmp_path / "ctx.db")
        with SnapshotDb(db_path) as database:
            conn = database.conn
        with pytest.raises(Exception):
            conn.execute("SELECT 1")

    def test_save_and_load_roundtrip(self, db):
        tracks = [_make_track_dict(track_id="tr_1"), _make_track_dict(track_id="tr_2")]
        db.save_tracks("pl_1", tracks)
        loaded = db.load_tracks("pl_1")
        assert len(loaded) == 2
        ids = {t['track_id'] for t in loaded}
        assert ids == {"tr_1", "tr_2"}

    def test_save_replaces_previous(self, db):
        db.save_tracks("pl_1", [_make_track_dict(track_id="tr_1")])
        db.save_tracks("pl_1", [_make_track_dict(track_id="tr_2")])
        loaded = db.load_tracks("pl_1")
        assert len(loaded) == 1
        assert loaded[0]['track_id'] == "tr_2"

    def test_load_empty_playlist(self, db):
        loaded = db.load_tracks("pl_nonexistent")
        assert loaded == []

    def test_playlists_are_isolated(self, db):
        db.save_tracks("pl_1", [_make_track_dict(track_id="tr_1", playlist_id="pl_1")])
        db.save_tracks("pl_2", [_make_track_dict(track_id="tr_2", playlist_id="pl_2")])
        assert len(db.load_tracks("pl_1")) == 1
        assert db.load_tracks("pl_1")[0]['track_id'] == "tr_1"
        assert len(db.load_tracks("pl_2")) == 1
        assert db.load_tracks("pl_2")[0]['track_id'] == "tr_2"

    def test_allows_duplicate_track_ids(self, db):
        tracks = [
            _make_track_dict(track_id="tr_1", added_at="t1"),
            _make_track_dict(track_id="tr_1", added_at="t2"),
        ]
        db.save_tracks("pl_1", tracks)
        loaded = db.load_tracks("pl_1")
        assert len(loaded) == 2

    def test_save_accepts_iterable(self, db):
        gen = (d for d in [_make_track_dict(track_id="tr_1")])
        db.save_tracks("pl_1", gen)
        loaded = db.load_tracks("pl_1")
        assert len(loaded) == 1

    def test_dump_csv(self, db, tmp_path):
        db.save_tracks("pl_1", [
            _make_track_dict(track_id="tr_1"),
            _make_track_dict(track_id="tr_2"),
        ])
        out = str(tmp_path / "dump.csv")
        count = db.dump_csv(out)
        assert count == 2
        content = (tmp_path / "dump.csv").read_text(encoding="utf-8")
        lines = content.strip().split("\n")
        assert lines[0].startswith("playlist_id,")
        assert len(lines) == 3

    def test_dump_csv_empty(self, db, tmp_path):
        out = str(tmp_path / "empty.csv")
        count = db.dump_csv(out)
        assert count == 0
        content = (tmp_path / "empty.csv").read_text(encoding="utf-8")
        lines = content.strip().split("\n")
        assert len(lines) == 1  # header only


# ---------------------------------------------------------------------------
# ChangeLogger
# ---------------------------------------------------------------------------

class TestChangeLogger:
    def test_creates_handlers(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cl = ChangeLogger()
        file_handlers = [
            h for h in cl.logger.handlers
            if isinstance(h, logging.FileHandler)
        ]
        console_handlers = [
            h for h in cl.logger.handlers
            if type(h) is logging.StreamHandler
        ]
        assert len(file_handlers) == 2
        assert len(console_handlers) == 1

    def test_run_and_changes_files_created(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        ChangeLogger()
        assert list(tmp_path.glob("run_*.log"))
        assert list(tmp_path.glob("changes_*.log"))

    def test_no_duplicate_handlers(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cl1 = ChangeLogger()
        count = len(cl1.logger.handlers)
        cl2 = ChangeLogger()
        assert len(cl2.logger.handlers) == count

    def test_info_delegates(self, tmp_path, monkeypatch, caplog):
        monkeypatch.chdir(tmp_path)
        cl = ChangeLogger()
        with caplog.at_level(logging.INFO, logger='spotify_snapshot'):
            cl.info("hello %s", "world")
        assert any("hello world" in r.message for r in caplog.records)

    def test_detects_additions(self, tmp_path, monkeypatch, caplog):
        monkeypatch.chdir(tmp_path)
        cl = ChangeLogger()
        with caplog.at_level(logging.INFO, logger='spotify_snapshot'):
            cl.log_changes([], [_make_track_dict()], "Test Playlist")
        assert any("[ADDED]" in r.message for r in caplog.records)
        assert any("Song One" in r.message for r in caplog.records)

    def test_detects_removals(self, tmp_path, monkeypatch, caplog):
        monkeypatch.chdir(tmp_path)
        cl = ChangeLogger()
        with caplog.at_level(logging.INFO, logger='spotify_snapshot'):
            cl.log_changes([_make_track_dict()], [], "Test Playlist")
        assert any("[REMOVED]" in r.message for r in caplog.records)

    def test_detects_property_change(self, tmp_path, monkeypatch, caplog):
        monkeypatch.chdir(tmp_path)
        cl = ChangeLogger()
        existing = [_make_track_dict(title="Old Title")]
        current = [_make_track_dict(title="New Title")]
        with caplog.at_level(logging.INFO, logger='spotify_snapshot'):
            cl.log_changes(existing, current, "Test Playlist")
        assert any("[CHANGED]" in r.message for r in caplog.records)
        assert any("Old Title" in r.message and "New Title" in r.message
                    for r in caplog.records)

    def test_no_log_when_unchanged(self, tmp_path, monkeypatch, caplog):
        monkeypatch.chdir(tmp_path)
        cl = ChangeLogger()
        track = _make_track_dict()
        with caplog.at_level(logging.INFO, logger='spotify_snapshot'):
            cl.log_changes([track.copy()], [track.copy()], "Test")
        change_records = [r for r in caplog.records if r.message.startswith("[")]
        assert len(change_records) == 0

    def test_multiple_field_changes(self, tmp_path, monkeypatch, caplog):
        monkeypatch.chdir(tmp_path)
        cl = ChangeLogger()
        existing = [_make_track_dict(title="Old", artist_name="Old Artist")]
        current = [_make_track_dict(title="New", artist_name="New Artist")]
        with caplog.at_level(logging.INFO, logger='spotify_snapshot'):
            cl.log_changes(existing, current, "Test Playlist")
        changed = [r for r in caplog.records if "[CHANGED]" in r.message]
        assert len(changed) == 2

    def test_duplicate_added(self, tmp_path, monkeypatch, caplog):
        monkeypatch.chdir(tmp_path)
        cl = ChangeLogger()
        existing = [_make_track_dict()]
        current = [
            _make_track_dict(added_at="2024-06-15_10:30:00"),
            _make_track_dict(added_at="2024-09-01_12:00:00"),
        ]
        with caplog.at_level(logging.INFO, logger='spotify_snapshot'):
            cl.log_changes(existing, current, "Test Playlist")
        added = [r for r in caplog.records if "[ADDED]" in r.message]
        assert len(added) == 1

    def test_duplicate_removed(self, tmp_path, monkeypatch, caplog):
        monkeypatch.chdir(tmp_path)
        cl = ChangeLogger()
        existing = [
            _make_track_dict(added_at="2024-06-15_10:30:00"),
            _make_track_dict(added_at="2024-09-01_12:00:00"),
        ]
        current = [_make_track_dict(added_at="2024-06-15_10:30:00")]
        with caplog.at_level(logging.INFO, logger='spotify_snapshot'):
            cl.log_changes(existing, current, "Test Playlist")
        removed = [r for r in caplog.records if "[REMOVED]" in r.message]
        assert len(removed) == 1

    def test_duplicate_no_false_changes(self, tmp_path, monkeypatch, caplog):
        monkeypatch.chdir(tmp_path)
        cl = ChangeLogger()
        tracks = [
            _make_track_dict(added_at="2024-06-15_10:30:00"),
            _make_track_dict(added_at="2024-09-01_12:00:00"),
        ]
        with caplog.at_level(logging.INFO, logger='spotify_snapshot'):
            cl.log_changes(
                [t.copy() for t in tracks],
                [t.copy() for t in tracks],
                "Test",
            )
        change_records = [r for r in caplog.records if r.message.startswith("[")]
        assert len(change_records) == 0

    def test_accepts_iterables(self, tmp_path, monkeypatch, caplog):
        monkeypatch.chdir(tmp_path)
        cl = ChangeLogger()
        with caplog.at_level(logging.INFO, logger='spotify_snapshot'):
            cl.log_changes(
                iter([]),
                iter([_make_track_dict()]),
                "Test",
            )
        assert any("[ADDED]" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# SpotifyFetcher._api_get  (retry behaviour)
# ---------------------------------------------------------------------------

def _mock_aiohttp_response(status, *, headers=None, json_data=None):
    """Build an object that works as an ``async with session.get(...)`` result."""
    resp = MagicMock()
    resp.status = status
    resp.headers = headers or {}
    resp.json = AsyncMock(return_value=json_data)
    resp.raise_for_status = MagicMock()
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=resp)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return ctx


class TestApiGet:
    async def test_returns_json_on_success(self, mock_spotify):
        mock_spotify._session.get = MagicMock(
            return_value=_mock_aiohttp_response(200, json_data={"ok": True}),
        )
        result = await mock_spotify._api_get("me/playlists")
        assert result == {"ok": True}

    async def test_retries_on_429_then_succeeds(self, mock_spotify):
        mock_spotify._session.get = MagicMock(side_effect=[
            _mock_aiohttp_response(429, headers={"Retry-After": "0"}),
            _mock_aiohttp_response(200, json_data={"ok": True}),
        ])
        result = await mock_spotify._api_get("me/playlists")
        assert result == {"ok": True}
        assert mock_spotify._session.get.call_count == 2

    async def test_raises_after_max_retries(self, mock_spotify):
        mock_spotify.retries = 2
        mock_spotify._session.get = MagicMock(
            return_value=_mock_aiohttp_response(
                429, headers={"Retry-After": "0"},
            ),
        )
        with pytest.raises(RuntimeError, match="rate limit"):
            await mock_spotify._api_get("me/playlists")
        assert mock_spotify._session.get.call_count == 3


# ---------------------------------------------------------------------------
# SpotifyFetcher.get_playlist_name_by_id
# ---------------------------------------------------------------------------

class TestGetPlaylistNameById:
    async def test_returns_cached_name(self, mock_spotify):
        mock_spotify.playlistmap = {"pl_1": "Chill Vibes"}
        assert await mock_spotify.get_playlist_name_by_id("pl_1") == "Chill Vibes"

    async def test_fetches_and_caches_when_missing(self, mock_spotify):
        mock_spotify._api_get = AsyncMock(return_value={"name": "Workout Mix"})
        name = await mock_spotify.get_playlist_name_by_id("pl_2")
        assert name == "Workout Mix"
        assert mock_spotify.playlistmap["pl_2"] == "Workout Mix"
        mock_spotify._api_get.assert_called_once_with(
            "playlists/pl_2", {"fields": "name"},
        )

    async def test_does_not_refetch_after_caching(self, mock_spotify):
        mock_spotify._api_get = AsyncMock(return_value={"name": "First Call"})
        await mock_spotify.get_playlist_name_by_id("pl_x")
        await mock_spotify.get_playlist_name_by_id("pl_x")
        mock_spotify._api_get.assert_called_once()


# ---------------------------------------------------------------------------
# SpotifyFetcher.get_all_playlists
# ---------------------------------------------------------------------------

class TestGetAllPlaylists:
    async def test_single_page(self, mock_spotify, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        mock_spotify._api_get = AsyncMock(
            return_value=FAKE_PLAYLIST_RESPONSE_PAGE1,
        )
        result = await mock_spotify.get_all_playlists()
        assert len(result) == 2
        assert mock_spotify.playlistmap["pl_1"] == "Chill Vibes"
        assert mock_spotify.playlistmap["pl_2"] == "Workout Mix"
        assert (tmp_path / "playlist.txt").exists()

    async def test_pagination(self, mock_spotify, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        full_page = {
            "items": [
                {"id": f"pl_{i}", "name": f"Playlist {i}"} for i in range(50)
            ],
            "total": 51,
        }
        partial_page = {
            "items": [{"id": "pl_last", "name": "Last Playlist"}],
        }
        mock_spotify._api_get = AsyncMock(
            side_effect=[full_page, partial_page],
        )
        result = await mock_spotify.get_all_playlists()
        assert len(result) == 51
        assert mock_spotify._api_get.call_count == 2

    async def test_playlist_txt_content(self, mock_spotify, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        mock_spotify._api_get = AsyncMock(
            return_value=FAKE_PLAYLIST_RESPONSE_PAGE1,
        )
        await mock_spotify.get_all_playlists()
        content = (tmp_path / "playlist.txt").read_text(encoding="utf-8")
        assert "pl_1: Chill Vibes\n" in content
        assert "pl_2: Workout Mix\n" in content


# ---------------------------------------------------------------------------
# SpotifyFetcher.get_tracks_in_one_playlist
# ---------------------------------------------------------------------------

class TestGetTracksInOnePlaylist:
    async def test_returns_correct_dicts(self, mock_spotify):
        mock_spotify._api_get = AsyncMock(return_value=FAKE_TRACK_RESPONSE)
        tracks = await mock_spotify.get_tracks_in_one_playlist("pl_1", "Chill Vibes")
        assert len(tracks) == 2
        t = tracks[0]
        assert t['playlist_id'] == "pl_1"
        assert t['playlist_name'] == "Chill Vibes"
        assert t['added_at'] == "2024-06-15_10:30:00"
        assert t['track_id'] == "tr_1"
        assert t['title'] == "Song One"
        assert t['disc_number'] == "1"
        assert t['track_number'] == "3"
        assert t['is_local'] == "False"
        assert t['album_id'] == "al_1"
        assert t['album_title'] == "Album One"
        assert t['artist_id'] == "ar_1, ar_2"
        assert t['artist_name'] == "Artist A, Artist B"

    async def test_skips_tracks_with_null_id(self, mock_spotify):
        mock_spotify._api_get = AsyncMock(
            return_value=FAKE_TRACK_RESPONSE_WITH_NULL,
        )
        tracks = await mock_spotify.get_tracks_in_one_playlist("pl_1", "Test")
        assert len(tracks) == 1
        assert tracks[0]['track_id'] == "tr_3"

    async def test_handles_duplicates(self, mock_spotify):
        mock_spotify._api_get = AsyncMock(
            return_value=FAKE_TRACK_RESPONSE_WITH_DUPLICATE,
        )
        tracks = await mock_spotify.get_tracks_in_one_playlist("pl_1", "Dupes")
        assert len(tracks) == 3
        assert sum(1 for t in tracks if t['track_id'] == "tr_1") == 2

    async def test_pagination(self, mock_spotify):
        full_page = {
            "items": [
                {
                    "added_at": f"2024-01-01T00:00:0{i % 10}Z",
                    "track": {
                        "id": f"tr_{i}",
                        "name": f"Track {i}",
                        "disc_number": 1,
                        "track_number": i,
                        "is_local": False,
                        "album": {"id": "al_1", "name": "Album"},
                        "artists": [{"id": "ar_1", "name": "Artist"}],
                    },
                }
                for i in range(50)
            ]
        }
        partial_page = {
            "items": [
                {
                    "added_at": "2024-02-01T00:00:00Z",
                    "track": {
                        "id": "tr_last",
                        "name": "Last Track",
                        "disc_number": 1,
                        "track_number": 1,
                        "is_local": False,
                        "album": {"id": "al_2", "name": "Album 2"},
                        "artists": [{"id": "ar_2", "name": "Artist 2"}],
                    },
                }
            ]
        }
        mock_spotify._api_get = AsyncMock(side_effect=[full_page, partial_page])
        tracks = await mock_spotify.get_tracks_in_one_playlist("pl_1", "Big Playlist")
        assert len(tracks) == 51


# ---------------------------------------------------------------------------
# SpotifyFetcher.get_playlist_items (integration-level)
# ---------------------------------------------------------------------------

class TestGetPlaylistItems:
    async def test_creates_db_and_stores_tracks(
        self, mock_spotify, tmp_path, monkeypatch,
    ):
        monkeypatch.chdir(tmp_path)
        mock_spotify.playlistmap = {"pl_1": "Chill Vibes"}
        mock_spotify._api_get = AsyncMock(return_value=FAKE_TRACK_RESPONSE)
        await mock_spotify.get_playlist_items(["pl_1"], [])

        conn = sqlite3.connect(str(tmp_path / "spotify_snapshot.db"))
        rows = conn.execute(
            "SELECT * FROM tracks WHERE playlist_id = 'pl_1'",
        ).fetchall()
        conn.close()
        assert len(rows) == 2

    async def test_stores_duplicate_tracks(
        self, mock_spotify, tmp_path, monkeypatch,
    ):
        monkeypatch.chdir(tmp_path)
        mock_spotify.playlistmap = {"pl_1": "Chill Vibes"}
        mock_spotify._api_get = AsyncMock(
            return_value=FAKE_TRACK_RESPONSE_WITH_DUPLICATE,
        )
        await mock_spotify.get_playlist_items(["pl_1"], [])

        conn = sqlite3.connect(str(tmp_path / "spotify_snapshot.db"))
        rows = conn.execute(
            "SELECT * FROM tracks WHERE playlist_id = 'pl_1'",
        ).fetchall()
        conn.close()
        assert len(rows) == 3

    async def test_skips_excluded_playlists(
        self, mock_spotify, tmp_path, monkeypatch,
    ):
        monkeypatch.chdir(tmp_path)
        mock_spotify.playlistmap = {
            "pl_1": "Chill Vibes", "pl_2": "Workout Mix",
        }
        mock_spotify._api_get = AsyncMock(return_value=FAKE_TRACK_RESPONSE)
        await mock_spotify.get_playlist_items(["pl_1", "pl_2"], ["pl_2"])

        conn = sqlite3.connect(str(tmp_path / "spotify_snapshot.db"))
        rows_pl1 = conn.execute(
            "SELECT * FROM tracks WHERE playlist_id = 'pl_1'",
        ).fetchall()
        rows_pl2 = conn.execute(
            "SELECT * FROM tracks WHERE playlist_id = 'pl_2'",
        ).fetchall()
        conn.close()
        assert len(rows_pl1) == 2
        assert len(rows_pl2) == 0

    async def test_logs_additions_on_first_run(
        self, mock_spotify, tmp_path, monkeypatch, caplog,
    ):
        monkeypatch.chdir(tmp_path)
        mock_spotify.playlistmap = {"pl_1": "Chill Vibes"}
        mock_spotify._api_get = AsyncMock(return_value=FAKE_TRACK_RESPONSE)
        with caplog.at_level(logging.INFO, logger='spotify_snapshot'):
            await mock_spotify.get_playlist_items(["pl_1"], [])
        added = [r for r in caplog.records if "[ADDED]" in r.message]
        assert len(added) == 2

    async def test_logs_removal_on_second_run(
        self, mock_spotify, tmp_path, monkeypatch, caplog,
    ):
        monkeypatch.chdir(tmp_path)
        mock_spotify.playlistmap = {"pl_1": "Chill Vibes"}

        mock_spotify._api_get = AsyncMock(return_value=FAKE_TRACK_RESPONSE)
        await mock_spotify.get_playlist_items(["pl_1"], [])

        one_track = {"items": [FAKE_TRACK_RESPONSE["items"][0]]}
        mock_spotify._api_get = AsyncMock(return_value=one_track)
        caplog.clear()
        with caplog.at_level(logging.INFO, logger='spotify_snapshot'):
            await mock_spotify.get_playlist_items(["pl_1"], [])
        removed = [r for r in caplog.records if "[REMOVED]" in r.message]
        assert len(removed) == 1
        assert "Song Two" in removed[0].message

    async def test_logs_change_on_second_run(
        self, mock_spotify, tmp_path, monkeypatch, caplog,
    ):
        monkeypatch.chdir(tmp_path)
        mock_spotify.playlistmap = {"pl_1": "Chill Vibes"}

        mock_spotify._api_get = AsyncMock(return_value=FAKE_TRACK_RESPONSE)
        await mock_spotify.get_playlist_items(["pl_1"], [])

        modified = {
            "items": [
                {
                    **FAKE_TRACK_RESPONSE["items"][0],
                    "track": {
                        **FAKE_TRACK_RESPONSE["items"][0]["track"],
                        "name": "Song One (Remastered)",
                    },
                },
                FAKE_TRACK_RESPONSE["items"][1],
            ]
        }
        mock_spotify._api_get = AsyncMock(return_value=modified)
        caplog.clear()
        with caplog.at_level(logging.INFO, logger='spotify_snapshot'):
            await mock_spotify.get_playlist_items(["pl_1"], [])
        changed = [r for r in caplog.records if "[CHANGED]" in r.message]
        assert len(changed) == 1
        assert "Song One (Remastered)" in changed[0].message

    async def test_creates_log_files(
        self, mock_spotify, tmp_path, monkeypatch,
    ):
        monkeypatch.chdir(tmp_path)
        mock_spotify.playlistmap = {"pl_1": "Chill Vibes"}
        mock_spotify._api_get = AsyncMock(return_value=FAKE_TRACK_RESPONSE)
        await mock_spotify.get_playlist_items(["pl_1"], [])
        assert list(tmp_path.glob("run_*.log"))
        assert list(tmp_path.glob("changes_*.log"))

    async def test_changes_file_only_has_changes(
        self, mock_spotify, tmp_path, monkeypatch,
    ):
        monkeypatch.chdir(tmp_path)
        mock_spotify.playlistmap = {"pl_1": "Chill Vibes"}
        mock_spotify._api_get = AsyncMock(return_value=FAKE_TRACK_RESPONSE)
        await mock_spotify.get_playlist_items(["pl_1"], [])
        changes_file = list(tmp_path.glob("changes_*.log"))[0]
        lines = changes_file.read_text(encoding="utf-8").strip().split("\n")
        for line in lines:
            assert "[ADDED]" in line or "[REMOVED]" in line or "[CHANGED]" in line

    async def test_run_file_has_all_messages(
        self, mock_spotify, tmp_path, monkeypatch,
    ):
        monkeypatch.chdir(tmp_path)
        mock_spotify.playlistmap = {"pl_1": "Chill Vibes"}
        mock_spotify._api_get = AsyncMock(return_value=FAKE_TRACK_RESPONSE)
        await mock_spotify.get_playlist_items(["pl_1"], [])
        run_file = list(tmp_path.glob("run_*.log"))[0]
        content = run_file.read_text(encoding="utf-8")
        assert "Fetching" in content
        assert "[ADDED]" in content
        assert "All playlists processed" in content

    async def test_logs_duration(
        self, mock_spotify, tmp_path, monkeypatch, caplog,
    ):
        monkeypatch.chdir(tmp_path)
        mock_spotify.playlistmap = {"pl_1": "Chill Vibes"}
        mock_spotify._api_get = AsyncMock(return_value=FAKE_TRACK_RESPONSE)
        with caplog.at_level(logging.INFO, logger='spotify_snapshot'):
            await mock_spotify.get_playlist_items(["pl_1"], [])
        final = [
            r for r in caplog.records
            if "All playlists processed" in r.message
        ]
        assert len(final) == 1
        import re
        assert re.search(r'\d+:\d{2}', final[0].message)

    async def test_accepts_iterable_playlists(
        self, mock_spotify, tmp_path, monkeypatch,
    ):
        monkeypatch.chdir(tmp_path)
        mock_spotify.playlistmap = {"pl_1": "Chill Vibes"}
        mock_spotify._api_get = AsyncMock(return_value=FAKE_TRACK_RESPONSE)
        await mock_spotify.get_playlist_items(iter(["pl_1"]), iter([]))

        conn = sqlite3.connect(str(tmp_path / "spotify_snapshot.db"))
        rows = conn.execute(
            "SELECT * FROM tracks WHERE playlist_id = 'pl_1'",
        ).fetchall()
        conn.close()
        assert len(rows) == 2


# ---------------------------------------------------------------------------
# get_configuration
# ---------------------------------------------------------------------------

class TestGetConfiguration:
    def test_reads_yaml_file(self, tmp_path):
        import yaml
        cfg = tmp_path / "test.yaml"
        config = {
            "client_id": "test_id",
            "secret": "test_secret",
            "redirect_uri": "http://localhost:9999",
            "playlists": ["pl_1", "pl_2"],
            "exclude": ["pl_2"],
        }
        cfg.write_text(yaml.dump(config), encoding="utf-8")
        client_id, secret, redirect_uri, playlists, exclude = get_configuration(str(cfg))
        assert client_id == "test_id"
        assert secret == "test_secret"
        assert redirect_uri == "http://localhost:9999"
        assert playlists == ["pl_1", "pl_2"]
        assert exclude == ["pl_2"]

    def test_returns_defaults_when_no_file(self, tmp_path):
        client_id, secret, redirect_uri, playlists, exclude = get_configuration(
            str(tmp_path / "nonexistent.yaml")
        )
        assert client_id == ""
        assert secret == ""
        assert redirect_uri == ""
        assert playlists == []
        assert exclude == []

    def test_partial_config(self, tmp_path):
        import yaml
        cfg = tmp_path / "partial.yaml"
        cfg.write_text(yaml.dump({"client_id": "only_id"}), encoding="utf-8")
        client_id, secret, redirect_uri, playlists, exclude = get_configuration(str(cfg))
        assert client_id == "only_id"
        assert secret == ""
        assert redirect_uri == ""
        assert playlists == []
        assert exclude == []

    def test_uses_default_filename(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        import yaml
        (tmp_path / CONFIG_FILENAME).write_text(
            yaml.dump({"client_id": "default_id", "secret": "default_secret"}),
            encoding="utf-8",
        )
        client_id, secret, _, _, _ = get_configuration()
        assert client_id == "default_id"


# ---------------------------------------------------------------------------
# get_arguments
# ---------------------------------------------------------------------------

class TestGetArguments:
    def test_defaults_config_to_constant(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["prog"])
        args = get_arguments()
        assert args.config == CONFIG_FILENAME
        assert args.concurrency == DEFAULT_CONCURRENCY
        assert args.retries == DEFAULT_RETRIES

    def test_custom_config_path(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["prog", "-f", "/tmp/my.yaml"])
        args = get_arguments()
        assert args.config == "/tmp/my.yaml"

    def test_parses_credential_args(self, monkeypatch):
        monkeypatch.setattr(
            "sys.argv", ["prog", "-i", "my_id", "-s", "my_secret"]
        )
        args = get_arguments()
        assert args.id == "my_id"
        assert args.secret == "my_secret"
        assert args.redirect_uri is None
        assert args.playlists is None
        assert args.excludes is None

    def test_dump_default_filename(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["prog", "-d"])
        args = get_arguments()
        assert args.dump == "snapshot_dump.csv"

    def test_dump_custom_filename(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["prog", "-d", "my_export.csv"])
        args = get_arguments()
        assert args.dump == "my_export.csv"

    def test_dump_not_set_by_default(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["prog"])
        args = get_arguments()
        assert args.dump is None

    def test_parses_redirect_uri(self, monkeypatch):
        monkeypatch.setattr(
            "sys.argv", ["prog", "-u", "http://localhost:9999"]
        )
        args = get_arguments()
        assert args.redirect_uri == "http://localhost:9999"

    def test_parses_concurrency(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["prog", "-n", "20"])
        args = get_arguments()
        assert args.concurrency == 20

    def test_parses_retries(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["prog", "-r", "5"])
        args = get_arguments()
        assert args.retries == 5

    def test_parses_all_args(self, monkeypatch):
        monkeypatch.setattr(
            "sys.argv",
            ["prog", "-f", "custom.yaml", "-i", "my_id", "-s", "my_secret",
             "-u", "http://localhost:9999",
             "-l", "pl_1", "pl_2", "-x", "pl_2",
             "-n", "15", "-r", "5"],
        )
        args = get_arguments()
        assert args.config == "custom.yaml"
        assert args.redirect_uri == "http://localhost:9999"
        assert args.playlists == ["pl_1", "pl_2"]
        assert args.excludes == ["pl_2"]
        assert args.concurrency == 15
        assert args.retries == 5


# ---------------------------------------------------------------------------
# Helpers for mocking async SpotifyFetcher in main() tests
# ---------------------------------------------------------------------------

def _make_mock_fetcher(**async_overrides):
    """Build a MagicMock that behaves as an async context manager."""
    mock_instance = MagicMock()
    mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
    mock_instance.__aexit__ = AsyncMock(return_value=False)
    mock_instance.get_playlist_items = AsyncMock()
    mock_instance.get_all_playlists = AsyncMock(return_value=[])
    for name, return_value in async_overrides.items():
        setattr(mock_instance, name, AsyncMock(return_value=return_value))
    return mock_instance


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

class TestMain:
    def test_uses_config_when_available(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("sys.argv", ["prog"])
        import yaml
        config = {
            "client_id": "cfg_id",
            "secret": "cfg_secret",
            "playlists": ["pl_1"],
            "exclude": [],
        }
        (tmp_path / CONFIG_FILENAME).write_text(
            yaml.dump(config), encoding="utf-8"
        )
        with patch("spotify_playlist_snapshot.snapshot.SpotifyFetcher") as MockSp:
            MockSp.return_value = _make_mock_fetcher()
            main()
            MockSp.assert_called_once_with(
                "cfg_id", "cfg_secret", DEFAULT_REDIRECT_URI,
                concurrency=DEFAULT_CONCURRENCY, retries=DEFAULT_RETRIES,
            )
            MockSp.return_value.get_playlist_items.assert_called_once_with(
                ["pl_1"], [],
            )

    def test_custom_config_path(self, tmp_path, monkeypatch):
        cfg = tmp_path / "custom.yaml"
        import yaml
        cfg.write_text(
            yaml.dump({"client_id": "cid", "secret": "sec",
                        "playlists": ["pl_x"]}),
            encoding="utf-8",
        )
        monkeypatch.setattr("sys.argv", ["prog", "-f", str(cfg)])
        with patch("spotify_playlist_snapshot.snapshot.SpotifyFetcher") as MockSp:
            MockSp.return_value = _make_mock_fetcher()
            main()
            MockSp.assert_called_once_with(
                "cid", "sec", DEFAULT_REDIRECT_URI,
                concurrency=DEFAULT_CONCURRENCY, retries=DEFAULT_RETRIES,
            )
            MockSp.return_value.get_playlist_items.assert_called_once_with(
                ["pl_x"], [],
            )

    def test_redirect_uri_from_config(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("sys.argv", ["prog"])
        import yaml
        config = {
            "client_id": "cid", "secret": "sec",
            "redirect_uri": "http://localhost:9999",
            "playlists": ["pl_1"],
        }
        (tmp_path / CONFIG_FILENAME).write_text(
            yaml.dump(config), encoding="utf-8"
        )
        with patch("spotify_playlist_snapshot.snapshot.SpotifyFetcher") as MockSp:
            MockSp.return_value = _make_mock_fetcher()
            main()
            MockSp.assert_called_once_with(
                "cid", "sec", "http://localhost:9999",
                concurrency=DEFAULT_CONCURRENCY, retries=DEFAULT_RETRIES,
            )

    def test_redirect_uri_cli_overrides_config(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        import yaml
        config = {
            "client_id": "cid", "secret": "sec",
            "redirect_uri": "http://localhost:9999",
            "playlists": ["pl_1"],
        }
        (tmp_path / CONFIG_FILENAME).write_text(
            yaml.dump(config), encoding="utf-8"
        )
        monkeypatch.setattr(
            "sys.argv", ["prog", "-u", "http://localhost:7777"],
        )
        with patch("spotify_playlist_snapshot.snapshot.SpotifyFetcher") as MockSp:
            MockSp.return_value = _make_mock_fetcher()
            main()
            MockSp.assert_called_once_with(
                "cid", "sec", "http://localhost:7777",
                concurrency=DEFAULT_CONCURRENCY, retries=DEFAULT_RETRIES,
            )

    def test_falls_back_to_cli_args(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "sys.argv", ["prog", "-i", "cli_id", "-s", "cli_secret",
                         "-l", "pl_a", "-x", "pl_b"]
        )
        with patch("spotify_playlist_snapshot.snapshot.SpotifyFetcher") as MockSp:
            MockSp.return_value = _make_mock_fetcher()
            main()
            MockSp.assert_called_once_with(
                "cli_id", "cli_secret", DEFAULT_REDIRECT_URI,
                concurrency=DEFAULT_CONCURRENCY, retries=DEFAULT_RETRIES,
            )
            MockSp.return_value.get_playlist_items.assert_called_once_with(
                ["pl_a"], ["pl_b"],
            )

    def test_cli_concurrency_and_retries(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "sys.argv", ["prog", "-i", "cid", "-s", "sec",
                         "-l", "pl_1", "-n", "5", "-r", "2"],
        )
        with patch("spotify_playlist_snapshot.snapshot.SpotifyFetcher") as MockSp:
            MockSp.return_value = _make_mock_fetcher()
            main()
            MockSp.assert_called_once_with(
                "cid", "sec", DEFAULT_REDIRECT_URI,
                concurrency=5, retries=2,
            )

    def test_dump_exports_and_exits(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with SnapshotDb() as sndb:
            sndb.save_tracks("pl_1", [
                _make_track_dict(track_id="tr_1"),
                _make_track_dict(track_id="tr_2"),
            ])
        monkeypatch.setattr("sys.argv", ["prog", "-d"])
        main()
        content = (tmp_path / "snapshot_dump.csv").read_text(encoding="utf-8")
        lines = content.strip().split("\n")
        assert len(lines) == 3

    def test_dump_custom_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with SnapshotDb() as sndb:
            sndb.save_tracks("pl_1", [_make_track_dict()])
        out = str(tmp_path / "custom.csv")
        monkeypatch.setattr("sys.argv", ["prog", "-d", out])
        main()
        assert (tmp_path / "custom.csv").exists()

    def test_exits_when_no_credentials(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("sys.argv", ["prog"])
        with pytest.raises(SystemExit):
            main()

    def test_fetches_all_playlists_when_none_specified(
        self, tmp_path, monkeypatch,
    ):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("sys.argv", ["prog"])
        import yaml
        config = {"client_id": "cfg_id", "secret": "cfg_secret"}
        (tmp_path / CONFIG_FILENAME).write_text(
            yaml.dump(config), encoding="utf-8",
        )
        with patch("spotify_playlist_snapshot.snapshot.SpotifyFetcher") as MockSp:
            mock_instance = _make_mock_fetcher(
                get_all_playlists=[{"id": "pl_auto1"}, {"id": "pl_auto2"}],
            )
            MockSp.return_value = mock_instance
            main()
            mock_instance.get_all_playlists.assert_called_once()
            mock_instance.get_playlist_items.assert_called_once_with(
                ["pl_auto1", "pl_auto2"], [],
            )

    def test_cli_overrides_config_playlists(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        import yaml
        config = {
            "client_id": "cid", "secret": "sec",
            "playlists": ["pl_cfg"],
        }
        (tmp_path / CONFIG_FILENAME).write_text(
            yaml.dump(config), encoding="utf-8"
        )
        monkeypatch.setattr("sys.argv", ["prog", "-l", "pl_cli"])
        with patch("spotify_playlist_snapshot.snapshot.SpotifyFetcher") as MockSp:
            MockSp.return_value = _make_mock_fetcher()
            main()
            MockSp.return_value.get_playlist_items.assert_called_once_with(
                ["pl_cli"], [],
            )
