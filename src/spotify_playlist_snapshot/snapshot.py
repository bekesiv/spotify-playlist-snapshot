"""Fetch all playlists information of a Spotify user and save it to a SQLite database."""
import asyncio
import csv
import sys
import os
import argparse
import logging
import sqlite3
import time
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime
from typing import Any

import aiohttp
import colorama
from colorama import Fore, Style
import yaml
from spotipy.oauth2 import SpotifyOAuth

CONFIG_FILENAME = 'configuration.yaml'
DB_FILENAME = 'spotify_snapshot.db'
DEFAULT_REDIRECT_URI = 'http://127.0.0.1:8000'
DEFAULT_CONCURRENCY = 10
DEFAULT_RETRIES = 8
MAX_RETRY_AFTER = 100


class RateLimitAbortError(RuntimeError):
    """Raised when Retry-After exceeds MAX_RETRY_AFTER seconds."""


class SnapshotDb:
    """Context-managed SQLite store for playlist track snapshots."""

    def __init__(self, db_path: str = DB_FILENAME) -> None:
        self.conn = sqlite3.connect(db_path)
        self._create_table()

    def _create_table(self) -> None:
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS tracks (
                playlist_id   TEXT NOT NULL,
                track_id      TEXT NOT NULL,
                playlist_name TEXT,
                added_at      TEXT,
                title         TEXT,
                disc_number   TEXT,
                track_number  TEXT,
                is_local      TEXT,
                album_id      TEXT,
                album_title   TEXT,
                artist_id     TEXT,
                artist_name   TEXT
            )
        """)
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tracks_playlist ON tracks(playlist_id)"
        )
        self.conn.commit()

    def load_tracks(self, playlist_id: str) -> list[dict[str, str]]:
        """Load all tracks for a playlist as a list of dicts."""
        cursor = self.conn.execute(
            "SELECT * FROM tracks WHERE playlist_id = ?", (playlist_id,)
        )
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor]

    def save_tracks(self, playlist_id: str, tracks: Iterable[dict[str, str]]) -> None:
        """Replace all tracks for a playlist with the current snapshot."""
        self.conn.execute("DELETE FROM tracks WHERE playlist_id = ?", (playlist_id,))
        for t in tracks:
            self.conn.execute(
                """INSERT INTO tracks
                   (playlist_id, track_id, playlist_name, added_at, title,
                    disc_number, track_number, is_local, album_id, album_title,
                    artist_id, artist_name)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (t['playlist_id'], t['track_id'], t['playlist_name'], t['added_at'],
                 t['title'], t['disc_number'], t['track_number'], t['is_local'],
                 t['album_id'], t['album_title'], t['artist_id'], t['artist_name']),
            )
        self.conn.commit()

    def dump_csv(self, output_path: str) -> int:
        """Export all tracks to a CSV file. Returns the number of rows written."""
        cursor = self.conn.execute("SELECT * FROM tracks ORDER BY playlist_name, added_at")
        columns = [desc[0] for desc in cursor.description]
        count = 0
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            for row in cursor:
                writer.writerow(row)
                count += 1
        return count

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> 'SnapshotDb':
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class _ChangesOnlyFilter(logging.Filter):
    """Pass only [ADDED] / [REMOVED] / [CHANGED] log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.msg.startswith('[')


class _ColoredConsoleFormatter(logging.Formatter):
    """Adds ANSI colors to console log lines based on message content."""

    _PREFIX_COLORS = {
        '[ADDED]': Fore.GREEN,
        '[REMOVED]': Fore.RED,
        '[CHANGED]': Fore.YELLOW,
    }

    def format(self, record: logging.LogRecord) -> str:
        formatted = super().format(record)
        for prefix, color in self._PREFIX_COLORS.items():
            if record.msg.startswith(prefix):
                return f"{color}{formatted}{Style.RESET_ALL}"
        if 'rate limit' in record.msg.lower():
            return f"{Fore.LIGHTBLACK_EX}{formatted}{Style.RESET_ALL}"
        return formatted


class ChangeLogger:
    """Triple-output logger (run file + changes file + console).

    The *run* log receives every message.  The *changes* log is a subset
    that only contains [ADDED], [REMOVED] and [CHANGED] entries.
    """

    TRACKED_FIELDS = [
        'playlist_name',
        'added_at',
        'title',
        'disc_number',
        'track_number',
        'is_local',
        'album_id',
        'album_title',
        'artist_id',
        'artist_name',
    ]

    def __init__(self) -> None:
        self._start_time = time.monotonic()
        self.logger = logging.getLogger('spotify_snapshot')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

            run_handler = logging.FileHandler(f'run_{timestamp}.log', encoding='utf-8')
            run_handler.setFormatter(formatter)

            changes_handler = logging.FileHandler(f'changes_{timestamp}.log', encoding='utf-8')
            changes_handler.setFormatter(formatter)
            changes_handler.addFilter(_ChangesOnlyFilter())

            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(
                _ColoredConsoleFormatter('%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
            )

            self.logger.addHandler(run_handler)
            self.logger.addHandler(changes_handler)
            self.logger.addHandler(console_handler)

    def elapsed_str(self) -> str:
        """Format elapsed time since construction as ``m:ss``."""
        elapsed = time.monotonic() - self._start_time
        minutes, seconds = divmod(int(elapsed), 60)
        return f"{minutes}:{seconds:02d}"

    def info(self, msg: str, *args: Any) -> None:
        self.logger.info(msg, *args)

    @staticmethod
    def _group_by_track_id(tracks: Iterable[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
        """Group a list of track dicts by track_id, preserving order."""
        groups: dict[str, list[dict[str, str]]] = defaultdict(list)
        for t in tracks:
            groups[t['track_id']].append(t)
        return groups

    def log_changes(
        self,
        existing_tracks: Iterable[dict[str, str]],
        current_tracks: Iterable[dict[str, str]],
        playlist_name: str,
    ) -> None:
        """Compare existing and current track lists, logging any differences.

        Tracks are grouped by track_id to handle duplicates correctly.
        Within each track_id group, entries are paired by added_at for
        stable comparison; extra entries are reported as added or removed.
        """
        existing = self._group_by_track_id(existing_tracks)
        current = self._group_by_track_id(current_tracks)
        all_ids = sorted(set(existing) | set(current))

        for track_id in all_ids:
            old_list = sorted(existing.get(track_id, []), key=lambda t: t.get('added_at', ''))
            new_list = sorted(current.get(track_id, []), key=lambda t: t.get('added_at', ''))
            paired = min(len(old_list), len(new_list))
            for prev, curr in zip(old_list[:paired], new_list[:paired]):
                for field in self.TRACKED_FIELDS:
                    old_val = str(prev.get(field, ''))
                    new_val = str(curr.get(field, ''))
                    if old_val != new_val:
                        self.logger.info(
                            '[CHANGED] "%s" (%s) in "%s": %s: "%s" -> "%s"',
                            curr['title'], track_id, playlist_name,
                            field, old_val, new_val,
                        )

            for t in new_list[paired:]:
                self.logger.info(
                    '[ADDED] "%s" by %s -> "%s"',
                    t['title'], t['artist_name'], playlist_name,
                )

            for t in old_list[paired:]:
                self.logger.info(
                    '[REMOVED] "%s" by %s <- "%s"',
                    t['title'], t['artist_name'], playlist_name,
                )


class SpotifyFetcher:
    """Async Spotify API client with concurrent fetching and rate-limit retry.

    Uses aiohttp for HTTP requests and spotipy's SpotifyOAuth for token
    management.  Must be used as an async context manager::

        async with SpotifyFetcher(client_id, secret) as fetcher:
            playlists = await fetcher.get_all_playlists()
    """

    BASE_URL = "https://api.spotify.com/v1"

    def __init__(self, client_id: str, secret: str,
                 redirect_uri: str = DEFAULT_REDIRECT_URI,
                 concurrency: int = DEFAULT_CONCURRENCY,
                 retries: int = DEFAULT_RETRIES) -> None:
        scope = "playlist-read-private playlist-read-collaborative"
        self.auth_manager = SpotifyOAuth(client_id=client_id, client_secret=secret, redirect_uri=redirect_uri, scope=scope)
        self.concurrency = concurrency
        self.retries = retries
        self.playlistmap: dict[str, str] = {}
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> 'SpotifyFetcher':
        self.auth_manager.get_access_token(as_dict=False)
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._session:
            await self._session.close()

    def _get_headers(self) -> dict[str, str]:
        token = self.auth_manager.get_access_token(as_dict=False)
        return {"Authorization": f"Bearer {token}"}

    async def _api_get(self, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """GET from the Spotify API with automatic retry on 429/529 rate limits."""
        url = f"{self.BASE_URL}/{endpoint}"
        logger = logging.getLogger('spotify_snapshot')

        for attempt in range(self.retries + 1):
            headers = self._get_headers()
            assert self._session is not None
            async with self._session.get(url, headers=headers, params=params) as resp:
                if resp.status in (429, 529):
                    retry_after = int(resp.headers.get('Retry-After', '1'))
                    if retry_after > MAX_RETRY_AFTER:
                        logger.warning(
                            'Rate limit on %s: Retry-After %ds exceeds %ds threshold — aborting',
                            endpoint, retry_after, MAX_RETRY_AFTER,
                        )
                        raise RateLimitAbortError(
                            f"Retry-After {retry_after}s on {endpoint} "
                            f"exceeds {MAX_RETRY_AFTER}s threshold"
                        )
                    if attempt < self.retries:
                        logger.warning(
                            'Rate limited on %s, retrying after %ds (attempt %d/%d)',
                            endpoint, retry_after,
                            attempt + 1, self.retries,
                        )
                        await asyncio.sleep(retry_after)
                        continue
                    raise RuntimeError(
                        f"Spotify API rate limit on {endpoint} "
                        f"after {self.retries} retries"
                    )
                resp.raise_for_status()
                return await resp.json()
        raise RuntimeError("Unreachable")

    async def get_playlist_name_by_id(self, playlist_id: str) -> str:
        """Retrieves the name of a playlist by its ID, using a local cache."""
        name = self.playlistmap.get(playlist_id)
        if name is None:
            data = await self._api_get(f"playlists/{playlist_id}", {"fields": "name"})
            name = data["name"]
            self.playlistmap[playlist_id] = name
        return name

    async def get_all_playlists(self) -> list[dict[str, Any]]:
        """Retrieves all playlists of the current user.

        The first page is fetched to discover the total count, then remaining
        pages are fetched concurrently.
        """
        first_page = await self._api_get("me/playlists", {"limit": 50, "offset": 0})
        playlists: list[dict[str, Any]] = list(first_page["items"])
        total = first_page.get("total", len(playlists))

        if total > 50:
            pages = await asyncio.gather(*(
                self._api_get("me/playlists", {"limit": 50, "offset": offset})
                for offset in range(50, total, 50)
            ))
            for page in pages:
                playlists.extend(page["items"])

        print("Your playlists:")
        with open('playlist.txt', 'w', encoding='utf-8') as f:
            for playlist in playlists:
                self.playlistmap[playlist["id"]] = playlist["name"]
                print(f'{playlist["id"]}: {playlist["name"]}')
                sys.stdout.flush()
                f.write(f'{playlist["id"]}: {playlist["name"]}\n')
        return playlists

    async def get_tracks_in_one_playlist(self, playlist_id: str, playlist_name: str) -> list[dict[str, str]]:
        """Retrieves all tracks in a playlist via sequential pagination."""
        tracks: list[dict[str, str]] = []
        offset = 0
        limit = 50
        fields = (
            'items(added_at, track(id, name, disc_number, track_number,'
            ' is_local, album(id, name), artists(id, name)))'
        )
        while True:
            response = await self._api_get(
                f"playlists/{playlist_id}/tracks",
                {"limit": limit, "offset": offset, "fields": fields},
            )
            for i in response["items"]:
                if i.get('track', {}).get('id') is None:
                    continue
                row = {
                    'playlist_id': playlist_id,
                    'playlist_name': playlist_name,
                    'added_at': i["added_at"].replace('T', '_').replace('Z', ''),
                    'track_id': i["track"]["id"],
                    'title': i["track"]["name"],
                    'disc_number': str(i["track"]["disc_number"]),
                    'track_number': str(i["track"]["track_number"]),
                    'is_local': str(i["track"]["is_local"]),
                    'album_id': i["track"]["album"]["id"],
                    'album_title': i["track"]["album"]["name"],
                    'artist_id': ", ".join(val["id"] for val in i["track"]["artists"]),
                    'artist_name': ", ".join(val["name"] for val in i["track"]["artists"]),
                }
                tracks.append(row)
            offset += limit
            if len(response["items"]) < limit:
                break
        return tracks

    async def get_playlist_items(self, playlists: Iterable[str], excludes: Iterable[str]) -> None:
        """Fetch tracks from multiple playlists concurrently, then update DB.

        Each playlist is fetched inside a semaphore-bounded task (see
        ``--concurrency``).  Within each task the pages of a single playlist
        are fetched sequentially so that we don't add two-dimensional
        concurrency.
        """
        changelog = ChangeLogger()
        exclude_set = set(excludes)
        semaphore = asyncio.Semaphore(self.concurrency)

        async def fetch_one(playlist_id: str) -> tuple[str, str, list[dict[str, str]]] | None:
            async with semaphore:
                playlist_name = await self.get_playlist_name_by_id(playlist_id)
                if playlist_id in exclude_set:
                    changelog.info('Skipping %s - %s', playlist_id, playlist_name)
                    return None
                changelog.info('Fetching %s: %s', playlist_id, playlist_name)
                tracks = await self.get_tracks_in_one_playlist(playlist_id, playlist_name)
                return playlist_id, playlist_name, tracks

        playlist_list = list(playlists)
        try:
            results = await asyncio.gather(*(fetch_one(pid) for pid in playlist_list))
        except RateLimitAbortError:
            changelog.info(
                'Aborted due to excessive rate limiting after %s',
                changelog.elapsed_str(),
            )
            raise

        with SnapshotDb() as db:
            for result in results:
                if result is None:
                    continue
                playlist_id, playlist_name, current_tracks = result
                existing = db.load_tracks(playlist_id)
                changelog.log_changes(existing, current_tracks, playlist_name)
                db.save_tracks(playlist_id, current_tracks)
        changelog.info(
            'All playlists processed in %s', changelog.elapsed_str(),
        )


def get_configuration(
    config_path: str = CONFIG_FILENAME,
) -> tuple[str, str, str, list[str], list[str]]:
    """Reads the configuration from a YAML file and returns the relevant settings."""
    data: dict[str, Any] = {}
    if os.path.exists(config_path):
        with open(config_path, encoding="utf-8", mode='r') as f:
            data = yaml.safe_load(f)
    return (data.get('client_id', ''),
            data.get('secret', ''),
            data.get('redirect_uri', ''),
            data.get('playlists', []),
            data.get('exclude', []))


def get_arguments() -> argparse.Namespace:
    """Parses command-line arguments for the Spotify Playlist Snapshot application."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--config', default=CONFIG_FILENAME, help='Path to configuration YAML file')
    parser.add_argument('-d', '--dump', nargs='?', const='snapshot_dump.csv', metavar='FILE',
                        help='Dump database contents to CSV and exit '
                             '(default: snapshot_dump.csv)')
    parser.add_argument('-i', '--id', help='Client ID to Spotify Web API')
    parser.add_argument('-s', '--secret', help='Secret to Spotify Web API')
    parser.add_argument('-u', '--redirect-uri', help='OAuth redirect URI (must match Spotify Dashboard)')
    parser.add_argument('-l', '--playlists', nargs='+', help='One or more playlist to retrieve')
    parser.add_argument('-x', '--excludes', nargs='+', help='One or more playlist to exclude')
    parser.add_argument('-n', '--concurrency', type=int, default=DEFAULT_CONCURRENCY,
                        help='Max concurrent playlist fetches '
                             f'(default: {DEFAULT_CONCURRENCY})')
    parser.add_argument('-r', '--retries', type=int, default=DEFAULT_RETRIES,
                        help='Max retries on 429 rate-limit responses '
                             f'(default: {DEFAULT_RETRIES})')
    return parser.parse_args()


async def async_main() -> None:
    """Async entry point: read config, authenticate, fetch and snapshot playlists."""
    args = get_arguments()

    if args.dump:
        with SnapshotDb() as db:
            count = db.dump_csv(args.dump)
        print(f"Exported {count} tracks to {args.dump}")
        return

    (client_id, secret, redirect_uri, playlists, excludes) = get_configuration(args.config)

    if not client_id:
        client_id = args.id or ''
    if not secret:
        secret = args.secret or ''
    if args.redirect_uri:
        redirect_uri = args.redirect_uri
    if not redirect_uri:
        redirect_uri = DEFAULT_REDIRECT_URI
    if args.playlists:
        playlists = args.playlists
    if args.excludes:
        excludes = args.excludes

    if not client_id or not secret:
        print("Error: client_id and secret must be provided "
              "via config file or -i/-s arguments", file=sys.stderr)
        sys.exit(1)

    try:
        async with SpotifyFetcher(client_id, secret, redirect_uri,
                                  concurrency=args.concurrency,
                                  retries=args.retries) as msp:
            if not playlists:
                playlists = [i["id"] for i in await msp.get_all_playlists()]
            await msp.get_playlist_items(playlists, excludes)
    except RateLimitAbortError:
        sys.exit(1)


def main() -> None:
    """Main entry point."""
    colorama.just_fix_windows_console()
    asyncio.run(async_main())
