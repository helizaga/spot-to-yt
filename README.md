# Spotify to YouTube Music Likes Transfer

Transfer your Spotify Liked Songs to YouTube Music by automatically liking matching tracks.

## Features

- ✅ Transfers all liked songs from Spotify to YouTube Music
- ✅ Likes each matched track on YouTube Music (adds to your Liked Songs)
- ✅ Smart matching with fuzzy title matching and duration checking
- ✅ Thread-safe concurrent processing for speed
- ✅ Progress tracking and resume capability
- ✅ Caching to avoid duplicate searches
- ✅ Generates detailed CSV report for unmatched songs
- ✅ Configurable rate limiting to avoid API throttling

## Prerequisites

- Python 3.7+
- Spotify Developer Account (free)
- YouTube Music account
- Google Cloud Console account (for YouTube OAuth)

## Setup Instructions

### 1. Clone and Install Dependencies

```bash
# Clone the repository or download the files
cd spot-to-yt

# Install required packages
pip3 install -r requirements.txt
```

### 2. Set Up Spotify API

1. Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
2. Click "Create App"
3. Fill in:
   - App Name: `Spotify to YT Music Transfer` (or any name)
   - App Description: `Personal use app for transferring liked songs`
   - Redirect URI: `http://127.0.0.1:8080`
4. Click "Create"
5. In your app settings, find:
   - Client ID
   - Client Secret (click "Show Client Secret")

### 3. Set Up YouTube Music OAuth

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the YouTube Data API v3
4. Create OAuth 2.0 credentials:
   - Application type: **TVs and Limited Input devices** (Important!)
   - Note: Do NOT select "Desktop app" - use "TVs and Limited Input devices"
5. Note down the Client ID and Client Secret

### 4. Configure Environment Variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your credentials:
# - SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET from Spotify Developer Dashboard
# - YTM_CLIENT_ID and YTM_CLIENT_SECRET from Google Cloud Console
```

**IMPORTANT**: Both YouTube Music credentials (YTM_CLIENT_ID and YTM_CLIENT_SECRET) are REQUIRED. The script will not run without them.

### 5. Authenticate YouTube Music

Run the YouTube Music OAuth setup (one-time only):

**Option 1: Using the setup script (recommended)**

```bash
python3 setup_youtube_oauth.py
```

**Option 2: Using ytmusicapi directly**

```bash
# If you have your credentials in .env:
ytmusicapi oauth --file oauth.json --client-id "$YTM_CLIENT_ID" --client-secret "$YTM_CLIENT_SECRET"

# Or provide them directly:
ytmusicapi oauth --file oauth.json --client-id YOUR_CLIENT_ID --client-secret YOUR_CLIENT_SECRET
```

This will:

1. Prompt for your Google API credentials (if not in .env when using the setup script)
2. Display a URL and code for authentication
3. You'll visit the URL, sign in with your Google/YouTube account
4. Enter the authorization code back in the terminal
5. Automatically save credentials to `oauth.json`

**Note**: The command uses the Google API flow for TV devices, so you'll need to manually enter a code rather than being redirected automatically.

### 6. Run the Transfer

```bash
python3 spotify_to_ytmusic_like_optimal.py
```

The script will:

1. Authenticate with Spotify (opens browser for first-time auth)
2. Fetch all your Spotify liked songs
3. Load your existing YouTube Music library to detect already liked songs
4. Search for each song on YouTube Music
5. Like each matched song (adds to YouTube Music Liked Songs)
6. Generate `unmatched_spotify_likes.csv` for songs that couldn't be found

**Note**: This script only likes songs on YouTube Music. It does NOT create or manage playlists.

## Configuration Options

Environment variables you can set in your `.env` file:

### Required

- `SPOTIFY_CLIENT_ID`: Your Spotify app client ID
- `SPOTIFY_CLIENT_SECRET`: Your Spotify app client secret
- `YTM_CLIENT_ID`: Your YouTube OAuth client ID
- `YTM_CLIENT_SECRET`: Your YouTube OAuth client secret

### Optional Performance Tuning

- `CONCURRENT_SEARCHES`: Number of parallel search threads (default: 10)
- `CONCURRENT_LIKES`: Number of parallel like threads (default: 5)
- `MAX_CONCURRENT_REQUESTS`: Max concurrent API requests (default: 5)
- `YTM_MIN_INTERVAL_MS`: Minimum milliseconds between API requests (default: 200)
- `YTM_OAUTH_PATH`: Path to YouTube Music OAuth file (default: "oauth.json")

## Output Files

- `oauth.json`: YouTube Music authentication (keep this secure!)
- `.cache-spotify-liked`: Spotify authentication cache
- `spotify_ytmusic_likes_cache.json`: Cache of matched songs and liked status
- `likes_progress.json`: Progress tracking for resume capability
- `unmatched_spotify_likes.csv`: Songs that couldn't be matched (includes album, Spotify URL, and track ID for manual matching)

## Features in Detail

### Smart Matching

- Uses fuzzy string matching for titles
- Checks artist names for matches
- Compares song duration (within 5-10 second tolerance)
- Tries multiple search strategies before giving up

### Thread Safety

- All cache operations are thread-safe
- Progress tracking prevents skipping songs on resume
- Concurrent API calls with proper rate limiting

### Resume Capability

If the script is interrupted, it will automatically resume from where it left off on the next run.

## Troubleshooting

### "YTM_CLIENT_ID and YTM_CLIENT_SECRET must be set"

These are now required. Get them from Google Cloud Console after setting up OAuth2 credentials.

### "YouTube Music OAuth file not found"

Run `python3 setup_youtube_oauth.py` or `ytmusicapi oauth` to authenticate. Make sure you have your YTM_CLIENT_ID and YTM_CLIENT_SECRET set.

### Songs not matching correctly

The script uses smart matching with duration checking. Check `unmatched_spotify_likes.csv` for songs to add manually. The CSV includes:

- Song title and artists
- Album name
- Spotify URL for easy reference
- Spotify track ID for exact identification

### Rate limiting errors

Adjust `YTM_MIN_INTERVAL_MS` in your `.env` file to increase the delay between requests (in milliseconds).

### Already liked songs

The script loads your YouTube Music library on startup to avoid re-liking songs. Songs already in your library will be skipped.

## Privacy & Security

- All authentication is handled locally
- OAuth tokens are stored only on your machine
- No data is sent to third parties
- **NEVER commit your `.env` file or `oauth.json` to version control**

## License

MIT License - Feel free to modify and use as needed.
