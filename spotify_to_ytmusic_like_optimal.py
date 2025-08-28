#!/usr/bin/env python3
"""
Like Spotify Liked Songs on YouTube Music - OPTIMAL VERSION
- Concurrent search AND liking for maximum speed
- Smart caching and progress persistence
- Better string matching for improved accuracy
- Rate limiting protection with exponential backoff
- Incremental processing for large libraries
"""

import os
import csv
import json
import time
import re
from typing import Optional, List, Dict, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
import threading
from pathlib import Path
import difflib
import requests

from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from spotipy.exceptions import SpotifyException
from ytmusicapi import YTMusic, OAuthCredentials

# Load environment variables
load_dotenv()

# Configuration
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
SPOTIFY_REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8080")
SPOTIFY_SCOPE = "user-library-read"

YTM_OAUTH_PATH = os.getenv("YTM_OAUTH_PATH", "oauth.json")
YTM_CLIENT_ID = os.getenv("YTM_CLIENT_ID")
YTM_CLIENT_SECRET = os.getenv("YTM_CLIENT_SECRET")

# Performance settings
CONCURRENT_SEARCHES = int(os.getenv("CONCURRENT_SEARCHES", "10"))  # Reduced for safety
CONCURRENT_LIKES = int(os.getenv("CONCURRENT_LIKES", "5"))  # Reduced for safety
BATCH_SIZE = 50  # Batch size for progress saves
MAX_CONCURRENT_REQUESTS = int(
    os.getenv("MAX_CONCURRENT_REQUESTS", "5")
)  # Limit concurrent API calls
MIN_REQUEST_INTERVAL = (
    float(os.getenv("YTM_MIN_INTERVAL_MS", "200")) / 1000
)  # Default 200ms

# Cache and progress files
CACHE_FILE = "spotify_ytmusic_likes_cache.json"
PROGRESS_FILE = "likes_progress.json"


def retry_with_backoff(max_retries=3):
    """Decorator for exponential backoff retry logic"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (
                    ConnectionError,
                    OSError,
                    KeyError,
                    TypeError,
                    requests.HTTPError,
                    requests.RequestException,
                ) as e:
                    if attempt < max_retries - 1:
                        wait_time = 2**attempt  # 1s, 2s, 4s
                        print(
                            f"  Retry {attempt + 1}/{max_retries} after {wait_time}s..."
                        )
                        time.sleep(wait_time)
                    else:
                        raise e
            return None

        return wrapper

    return decorator


class ThreadSafeYTMusic:
    """Thread-safe wrapper for YTMusic client with rate limiting"""

    def __init__(self, ytm: YTMusic):
        self.ytm = ytm
        self.semaphore = threading.BoundedSemaphore(MAX_CONCURRENT_REQUESTS)
        self.rate_lock = threading.Lock()
        self.last_request_time = 0
        self.min_request_interval = MIN_REQUEST_INTERVAL

    def _throttle(self):
        """Ensure minimum time between requests"""
        with self.rate_lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.min_request_interval:
                time.sleep(self.min_request_interval - time_since_last)
            self.last_request_time = time.time()

    @retry_with_backoff(max_retries=3)
    def search(self, query: str, filter_type: str = None, limit: int = None):
        """Thread-safe search with rate limiting and retry"""
        with self.semaphore:
            self._throttle()
            return self.ytm.search(query, filter=filter_type, limit=limit)

    @retry_with_backoff(max_retries=3)
    def rate_song(self, video_id: str, rating: str):
        """Thread-safe rate song with rate limiting and retry"""
        with self.semaphore:
            self._throttle()
            return self.ytm.rate_song(video_id, rating)

    @retry_with_backoff(max_retries=3)
    def get_library_songs(self, limit: int = None):
        """Get already liked songs from library"""
        with self.semaphore:
            self._throttle()
            return self.ytm.get_library_songs(limit=limit)


class CacheManager:
    """Enhanced cache manager with progress tracking"""

    def __init__(
        self, cache_file: str = CACHE_FILE, progress_file: str = PROGRESS_FILE
    ):
        self.lock = threading.RLock()  # Add thread safety
        self.cache_file = cache_file
        self.progress_file = progress_file
        self.cache = self.load_cache()
        self.liked_songs = self.cache.get("liked_songs", set())
        if isinstance(self.liked_songs, list):
            self.liked_songs = set(self.liked_songs)
        self.progress = self.load_progress()
        self.completed_indices = set()  # Track completed work
        self.last_save_time = time.time()

    def load_cache(self) -> Dict:
        """Load cache from disk"""
        if Path(self.cache_file).exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    cache = json.load(f)
                    print(
                        f"📦 Loaded cache with {len(cache.get('matches', {}))} cached matches"
                    )
                    print(
                        f"   and {len(cache.get('liked_songs', []))} already liked songs"
                    )
                    return cache
            except (json.JSONDecodeError, OSError, IOError):
                return {"matches": {}, "liked_songs": []}
        return {"matches": {}, "liked_songs": []}

    def load_progress(self) -> Dict:
        """Load progress tracking"""
        if Path(self.progress_file).exists():
            try:
                with open(self.progress_file, "r", encoding="utf-8") as f:
                    progress = json.load(f)
                    processed = progress.get("processed", 0)
                    print(f"📈 Resuming from previous run: {processed} songs processed")
                    return progress
            except (json.JSONDecodeError, OSError, IOError):
                return {}
        return {}

    def save_cache(self, force=False):
        """Save cache to disk with rate limiting"""
        with self.lock:
            current_time = time.time()
            if force or (
                current_time - self.last_save_time > 30
            ):  # Save every 30 seconds
                self.cache["liked_songs"] = list(self.liked_songs)
                with open(self.cache_file, "w", encoding="utf-8") as f:
                    json.dump(self.cache, f, indent=2)
                self.last_save_time = current_time

    def save_progress(self, processed: int, total: int):
        """Save progress for resuming"""
        with self.lock:
            self.progress = {
                "processed": processed,
                "total": total,
                "timestamp": time.time(),
            }
            with open(self.progress_file, "w", encoding="utf-8") as f:
                json.dump(self.progress, f, indent=2)

    def mark_completed(self, index: int) -> int:
        """Mark an index as completed and return the new contiguous processed count"""
        with self.lock:
            self.completed_indices.add(index)
            # Find largest contiguous completed index
            processed = 0
            while processed in self.completed_indices:
                processed += 1
            return processed

    def clear_progress(self):
        """Clear progress file after completion"""
        if Path(self.progress_file).exists():
            os.remove(self.progress_file)

    def get_spotify_key(
        self, title: str, artists: str, album: str, track_id: str = None
    ) -> str:
        """Generate unique key for a Spotify track"""
        # Prefer track ID if available for stability
        if track_id:
            return f"spotify:{track_id}"
        return f"{title}|||{artists}|||{album}"

    def get_cached_video_id(
        self, title: str, artists: str, album: str, track_id: str = None
    ) -> Optional[str]:
        """Get cached YouTube video ID for a Spotify track"""
        with self.lock:
            key = self.get_spotify_key(title, artists, album, track_id)
            return self.cache.setdefault("matches", {}).get(key)

    def add_to_cache(
        self,
        title: str,
        artists: str,
        album: str,
        video_id: str,
        track_id: str = None,
        metadata: dict = None,
    ):
        """Add a match to the cache with metadata"""
        if video_id:
            with self.lock:
                key = self.get_spotify_key(title, artists, album, track_id)
                self.cache.setdefault("matches", {})[key] = {
                    "video_id": video_id,
                    "metadata": metadata or {},
                }

    def is_already_liked(self, video_id: str) -> bool:
        """Check if a song was already liked"""
        with self.lock:
            return video_id in self.liked_songs

    def mark_as_liked(self, video_id: str):
        """Mark a song as liked"""
        with self.lock:
            self.liked_songs.add(video_id)

    def set_initial_liked_songs(self, liked_songs: Set[str]):
        """Set initial liked songs from YouTube Music library"""
        with self.lock:
            self.liked_songs = liked_songs


def normalize_string(s: str) -> str:
    """Normalize string for better matching"""
    # Remove special characters, convert to lowercase, strip whitespace
    s = re.sub(r"[^\w\s]", "", s.lower())
    s = re.sub(r"\s+", " ", s).strip()
    return s


def smart_match(
    title: str, artists: str, result: dict, spotify_duration_ms: int = None
) -> float:
    """Smart matching with normalization and scoring
    Returns a match score from 0 to 1, where 1 is perfect match
    """
    r_title = normalize_string(result.get("title", ""))
    r_artists = normalize_string(
        " ".join([a.get("name", "") for a in result.get("artists", [])])
    )

    norm_title = normalize_string(title)
    norm_artists = [normalize_string(a.strip()) for a in artists.split(",")]

    # Title matching with fuzzy matching
    title_ratio = difflib.SequenceMatcher(None, norm_title, r_title).ratio()
    title_match = title_ratio > 0.8 or norm_title in r_title or r_title in norm_title

    # Artist matching - at least one artist should match
    artist_match = False
    artist_exact_match = False
    for artist in norm_artists:
        if artist in r_artists:
            artist_match = True
            if artist == r_artists or r_artists in artist:
                artist_exact_match = True
                break

    # Duration matching if available
    duration_score = 1.0
    if spotify_duration_ms and result.get("duration_seconds"):
        spotify_duration_s = spotify_duration_ms / 1000
        yt_duration = result.get("duration_seconds")
        duration_diff = abs(spotify_duration_s - yt_duration)
        if duration_diff <= 5:  # Within 5 seconds
            duration_score = 1.0
        elif duration_diff <= 10:
            duration_score = 0.8
        else:
            duration_score = max(
                0.5, 1 - duration_diff / 60
            )  # Penalty for large differences

    if not title_match or not artist_match:
        return 0

    # Calculate combined score
    score = (
        0.4 * title_ratio
        + 0.3 * (1.0 if artist_exact_match else 0.7)
        + 0.3 * duration_score
    )
    return score


def spotify_client():
    """Initialize Spotify client"""
    return spotipy.Spotify(
        auth_manager=SpotifyOAuth(
            client_id=SPOTIFY_CLIENT_ID,
            client_secret=SPOTIFY_CLIENT_SECRET,
            redirect_uri=SPOTIFY_REDIRECT_URI,
            scope=SPOTIFY_SCOPE,
            open_browser=True,
            cache_path=".cache-spotify-liked",
        )
    )


def fetch_all_liked(sp: spotipy.Spotify) -> List[Dict[str, any]]:
    """Fetch all liked songs from Spotify"""
    print("🎵 Fetching liked songs from Spotify...")
    out = []

    try:
        results = sp.current_user_saved_tracks(limit=50)
    except SpotifyException as e:
        print(f"Error fetching Spotify liked songs: {e}")
        return []

    while True:
        for item in results["items"]:
            added_at = item["added_at"]
            tr = item["track"]
            if tr:  # Some tracks might be None
                out.append(
                    {
                        "added_at": added_at,
                        "title": tr["name"],
                        "artists": ", ".join(a["name"] for a in tr["artists"]),
                        "album": tr["album"]["name"],
                        "url": tr.get("external_urls", {}).get("spotify", ""),
                        "track_id": tr.get("id", ""),
                        "duration_ms": tr.get("duration_ms", 0),
                    }
                )

        if results.get("next"):
            try:
                results = sp.next(results)
            except SpotifyException as e:
                print(f"Error fetching next page: {e}")
                break
        else:
            break

    print(f"✅ Found {len(out)} liked songs on Spotify")
    return out


def search_song(
    ytm_safe: ThreadSafeYTMusic,
    song_data: Dict[str, any],
    cache_manager: CacheManager,
) -> Dict:
    """Enhanced search with smart matching"""
    title = song_data["title"]
    artists = song_data["artists"]
    album = song_data["album"]
    track_id = song_data.get("track_id")
    duration_ms = song_data.get("duration_ms")
    url = song_data.get("url", "")

    # Check cache first
    cached = cache_manager.get_cached_video_id(title, artists, album, track_id)
    if cached:
        video_id = cached["video_id"] if isinstance(cached, dict) else cached
        return {
            "title": title,
            "artists": artists,
            "album": album,
            "video_id": video_id,
            "spotify_url": url,
            "track_id": track_id,
            "from_cache": True,
            "already_liked": cache_manager.is_already_liked(video_id),
        }

    # Smarter search queries - most specific first
    queries = [
        f"{title} {artists} {album}",  # Most specific
        f"{title} {artists}",
        f"{title} {artists} official audio",
    ]

    video_id = None
    for q_idx, q in enumerate(queries):
        try:
            # Use more results for better matching
            results = ytm_safe.search(q, filter_type="songs", limit=10) or []

            # Smart matching with scoring
            best_match = None
            best_score = 0
            for r in results:
                score = smart_match(title, artists, r, duration_ms)
                if score > best_score:
                    best_score = score
                    best_match = r

            # Use best match if score is good enough
            if best_match and best_score > 0.7:
                video_id = best_match.get("videoId")
                break

            # Fallback to first result only on last query if score is reasonable
            if (
                not video_id
                and results
                and q_idx == len(queries) - 1
                and best_score > 0.5
            ):
                video_id = results[0].get("videoId")

            if video_id:
                break

        except (
            ConnectionError,
            OSError,
            KeyError,
            TypeError,
            requests.HTTPError,
            requests.RequestException,
        ):
            continue  # Try next query

    # Cache the result
    if video_id:
        cache_manager.add_to_cache(title, artists, album, video_id, track_id)

    return {
        "title": title,
        "artists": artists,
        "album": album,
        "video_id": video_id,
        "spotify_url": url,
        "track_id": track_id,
        "from_cache": False,
        "already_liked": (
            cache_manager.is_already_liked(video_id) if video_id else False
        ),
    }


def like_single_song(
    ytm_safe: ThreadSafeYTMusic, song: Dict, cache_manager: CacheManager
) -> bool:
    """Like a single song with error handling"""
    try:
        ytm_safe.rate_song(song["video_id"], "LIKE")
        cache_manager.mark_as_liked(song["video_id"])
        return True
    except (
        ConnectionError,
        OSError,
        KeyError,
        TypeError,
        requests.HTTPError,
        requests.RequestException,
    ) as e:
        print(f"  Failed to like '{song['title']}': {e}")
        return False


def like_songs_concurrent(
    ytm_safe: ThreadSafeYTMusic, songs_to_like: List[Dict], cache_manager: CacheManager
) -> int:
    """Like songs concurrently for maximum speed"""
    if not songs_to_like:
        return 0

    print(f"\n👍 Liking {len(songs_to_like)} songs concurrently...")
    liked_count = 0

    with ThreadPoolExecutor(max_workers=CONCURRENT_LIKES) as executor:
        # Submit all like tasks
        future_to_song = {
            executor.submit(like_single_song, ytm_safe, song, cache_manager): song
            for song in songs_to_like
            if song["video_id"] and not song["already_liked"]
        }

        # Process results as they complete
        completed = 0
        for future in as_completed(future_to_song):
            completed += 1
            if future.result():
                liked_count += 1

            # Progress update
            if completed % 50 == 0:
                print(f"  Liked {completed}/{len(songs_to_like)} songs...")
                cache_manager.save_cache()

    return liked_count


def load_ytm_library(ytm_safe: ThreadSafeYTMusic) -> Set[str]:
    """Load already liked songs from YouTube Music library"""
    print("📚 Loading YouTube Music library to detect already liked songs...")
    liked_video_ids = set()
    try:
        library_songs = ytm_safe.get_library_songs(limit=5000) or []
        for song in library_songs:
            if song.get("videoId"):
                liked_video_ids.add(song["videoId"])
        print(f"   Found {len(liked_video_ids)} songs already in YouTube Music library")
    except (
        ConnectionError,
        OSError,
        KeyError,
        TypeError,
        requests.HTTPError,
        requests.RequestException,
        ValueError,
    ) as e:
        print(f"   Could not load library (will rely on cache): {e}")
    return liked_video_ids


def main():
    """Main execution with all optimizations"""
    # Check YouTube Music credentials FIRST
    if not YTM_CLIENT_ID or not YTM_CLIENT_SECRET:
        print("❌ Error: YTM_CLIENT_ID and YTM_CLIENT_SECRET must be set")
        print("   Please add them to your .env file or export as environment variables")
        print(
            "   Get these from Google Cloud Console after setting up OAuth2 credentials"
        )
        return

    # Check Spotify credentials
    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
        print("❌ Error: SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET must be set")
        return

    # Initialize Spotify
    sp = spotify_client()
    print("✅ Spotify authentication successful")

    # Initialize YouTube Music
    if not os.path.exists(YTM_OAUTH_PATH):
        print(f"❌ YouTube Music OAuth file not found: {YTM_OAUTH_PATH}")
        print("Please run: python3 setup_youtube_oauth.py")
        return

    try:
        oauth_credentials = OAuthCredentials(
            client_id=YTM_CLIENT_ID, client_secret=YTM_CLIENT_SECRET
        )
        ytm = YTMusic(YTM_OAUTH_PATH, oauth_credentials=oauth_credentials)
        ytm_safe = ThreadSafeYTMusic(ytm)
    except (FileNotFoundError, ValueError, OSError, KeyError) as e:
        print(f"❌ Error initializing YouTube Music: {e}")
        return

    print("✅ YouTube Music authentication successful")

    # Initialize cache
    cache_manager = CacheManager()

    # Load YouTube Music library to detect already liked songs
    ytm_liked = load_ytm_library(ytm_safe)
    if ytm_liked:
        cache_manager.set_initial_liked_songs(ytm_liked)

    # Fetch Spotify liked songs
    liked = fetch_all_liked(sp)
    if not liked:
        print("❌ No liked songs found")
        return

    print(f"📊 Processing {len(liked)} Spotify liked songs")

    # Check if resuming from previous run
    start_idx = cache_manager.progress.get("processed", 0)
    if start_idx > 0:
        print(f"♻️  Resuming from song {start_idx}")
        liked = liked[start_idx:]

    # Search for all songs concurrently
    print("\n🔍 Searching for songs on YouTube Music...")
    start_time = time.time()

    matched_songs = []
    unmatched_songs = []
    cache_hits = 0
    already_liked = 0

    with ThreadPoolExecutor(max_workers=CONCURRENT_SEARCHES) as executor:
        # Submit all search tasks
        future_to_idx = {
            executor.submit(search_song, ytm_safe, song_data, cache_manager): (
                idx + start_idx,
                song_data,
            )
            for idx, song_data in enumerate(liked)
        }

        # Process results as they complete
        completed = 0
        for future in as_completed(future_to_idx):
            idx, _ = future_to_idx[future]
            result = future.result()
            completed += 1

            if result["from_cache"]:
                cache_hits += 1

            if result["video_id"]:
                matched_songs.append(result)
                if result["already_liked"]:
                    already_liked += 1
            else:
                unmatched_songs.append(result)

            # Track proper progress
            actual_processed = cache_manager.mark_completed(idx)

            # Save progress periodically
            if completed % BATCH_SIZE == 0:
                cache_manager.save_cache()
                cache_manager.save_progress(actual_processed, len(liked) + start_idx)
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                print(f"  Progress: {completed}/{len(liked)} ({rate:.1f} songs/sec)")
                print(f"    Cache hits: {cache_hits}, Already liked: {already_liked}")

    search_time = time.time() - start_time
    songs_per_sec = len(liked) / search_time if search_time > 0 else 0
    print(
        f"\n✅ Search completed in {search_time:.1f} seconds ({songs_per_sec:.1f} songs/sec)"
    )

    # Filter songs that need to be liked
    songs_to_like = [s for s in matched_songs if not s["already_liked"]]

    # Like songs concurrently
    like_start_time = time.time()
    total_liked = like_songs_concurrent(ytm_safe, songs_to_like, cache_manager)
    like_time = time.time() - like_start_time

    if songs_to_like:
        like_rate = total_liked / like_time if like_time > 0 else 0
        print(
            f"✅ Liked {total_liked} songs in {like_time:.1f} seconds ({like_rate:.1f} songs/sec)"
        )

    # Save final cache and clear progress
    cache_manager.save_cache(force=True)
    cache_manager.clear_progress()

    # Write unmatched songs to CSV with more metadata
    if unmatched_songs:
        fname = "unmatched_spotify_likes.csv"
        with open(fname, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["title", "artists", "album", "spotify_url", "track_id"])
            for song in unmatched_songs:
                w.writerow(
                    [
                        song["title"],
                        song["artists"],
                        song.get("album", ""),
                        song.get("spotify_url", ""),
                        song.get("track_id", ""),
                    ]
                )
        print(f"\n📄 Wrote {len(unmatched_songs)} unmatched songs to {fname}")

    # Final summary
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("🎉 Like Transfer Complete!")
    print("=" * 60)
    print(f"⏱️  Total time: {total_time:.1f} seconds")
    print(f"📊 Overall performance: {len(liked)/total_time:.1f} songs/sec")
    print(f"🔍 Search phase: {search_time:.1f}s ({songs_per_sec:.1f} songs/sec)")
    if songs_to_like:
        print(f"👍 Like phase: {like_time:.1f}s ({like_rate:.1f} songs/sec)")
    print(
        f"💾 Cache efficiency: {cache_hits}/{len(liked)} ({100*cache_hits/len(liked):.1f}%)"
    )
    print(f"✅ Matched songs: {len(matched_songs)}")
    print(f"🆕 Newly liked: {total_liked}")
    print(f"♻️  Already liked: {already_liked}")
    print(f"❌ Unmatched: {len(unmatched_songs)}")
    print("=" * 60)
    print("\n💡 This optimized version uses:")
    print(f"   • Concurrent searching ({CONCURRENT_SEARCHES} threads)")
    print(f"   • Concurrent liking ({CONCURRENT_LIKES} threads)")
    print(f"   • Max {MAX_CONCURRENT_REQUESTS} concurrent API requests")
    print("   • Smart matching with duration checking")
    print("   • Thread-safe caching and progress tracking")
    print("   • Progress persistence (resume if interrupted)")
    print("   • Exponential backoff retry")
    print(
        f"   • Rate limiting protection ({MIN_REQUEST_INTERVAL*1000:.0f}ms between requests)"
    )


if __name__ == "__main__":
    main()
