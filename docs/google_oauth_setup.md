# Google OAuth Credentials For BirdVision

BirdVision uses Google OAuth 2.0 for web login when auth is enabled. You need
three required values:

- `GOOGLE_CLIENT_ID`
- `GOOGLE_CLIENT_SECRET`
- `SESSION_SECRET`

And, for deployments behind HTTPS reverse proxies, usually a fourth value:

- `GOOGLE_REDIRECT_URI` or `auth.redirect_uri`

The Google values come from a Google Cloud OAuth client. The session secret is
generated locally by you.

## 1. Create Or Pick A Google Cloud Project

1. Go to Google Cloud Console.
2. Open the project picker.
3. Create a new project or select an existing one for BirdVision.

Use a project dedicated to BirdVision if you want cleaner separation from other
Google API work.

## 2. Configure The OAuth Consent Screen

1. In the Google Cloud Console, go to `APIs & Services`.
2. Open `OAuth consent screen`.
3. Choose `External` unless you are restricting access to a Google Workspace
   organization and specifically want `Internal`.
4. Fill in the required app details:
   - App name: `BirdVision` or your deployment name
   - User support email
   - Developer contact email
5. Save the consent screen configuration.

If the app is left in testing mode, add the Google accounts that should be able
to sign in as test users.

## 3. Create The OAuth Client

1. Go to `APIs & Services` > `Credentials`.
2. Click `Create credentials`.
3. Choose `OAuth client ID`.
4. For application type, choose `Web application`.
5. Give it a name such as `BirdVision Web`.
6. Add authorized redirect URIs for every BirdVision host that will use login.

Typical redirect URIs:

- `http://localhost:3587/auth/callback`
- `https://birdvision.example.com/auth/callback`

The redirect URI must exactly match the URL BirdVision uses for the callback.
Wrong scheme, host, port, or path will break login.

If BirdVision is behind a reverse proxy or TLS terminator, set an explicit
redirect URI in BirdVision instead of relying on the incoming request URL.
Otherwise the app may generate `http://.../auth/callback` internally while
Google only allows `https://.../auth/callback`.

## 4. Copy The Required Google Credentials

After creating the OAuth client, Google shows:

- Client ID
- Client secret

Use those values as:

```bash
export GOOGLE_CLIENT_ID="your-client-id.apps.googleusercontent.com"
export GOOGLE_CLIENT_SECRET="your-client-secret"
```

You can also place them under `auth.google_client_id` and
`auth.google_client_secret` in `config.yaml`, but environment variables are the
safer default.

## 5. Set The Redirect URI BirdVision Should Use

BirdVision can either infer the callback URL from the current request or use an
explicit configured value. For reverse-proxied HTTPS deployments, the explicit
value is safer.

`config.yaml` example:

```yaml
auth:
  google_client_id: "your-client-id.apps.googleusercontent.com"
  google_client_secret: "your-client-secret"
  redirect_uri: "https://birdvision.example.com/auth/callback"
  session_secret: "paste-generated-secret-here"
  allowed_emails:
    - you@example.com
```

Environment variable equivalent:

```bash
export GOOGLE_REDIRECT_URI="https://birdvision.example.com/auth/callback"
```

## 6. Generate A Session Secret

BirdVision also needs a local secret for signing session cookies:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

Set the result as:

```bash
export SESSION_SECRET="paste-generated-secret-here"
```

## 7. Allow Specific Uploaders

Google login only identifies the user. BirdVision separately checks whether the
signed-in email is allowed to upload or reprocess jobs.

Add allowed addresses in `config.yaml`:

```yaml
auth:
  allowed_emails:
    - you@example.com
    - teammate@example.com
```

`allowed_emails` is re-read from `config.yaml` at request time, so adding or
removing a user takes effect without restarting the server.

## 8. Run BirdVision

Docker example:

```bash
export GOOGLE_CLIENT_ID="your-client-id.apps.googleusercontent.com"
export GOOGLE_CLIENT_SECRET="your-client-secret"
export SESSION_SECRET="$(python -c 'import secrets; print(secrets.token_hex(32))')"
docker compose up
```

For Docker deployments behind HTTPS, set `auth.redirect_uri` in `config.yaml`
to the exact public callback URL registered in Google Cloud. That is the
simplest path because `config.yaml` is already bind-mounted into the container.

Local `uv` example:

```bash
export GOOGLE_CLIENT_ID="your-client-id.apps.googleusercontent.com"
export GOOGLE_CLIENT_SECRET="your-client-secret"
export GOOGLE_REDIRECT_URI="https://birdvision.example.com/auth/callback"
export SESSION_SECRET="$(python -c 'import secrets; print(secrets.token_hex(32))')"
uv run scripts/serve.py
```

For local development without auth, use:

```bash
uv run scripts/serve.py --debug
```

## Common Failure Modes

- `redirect_uri_mismatch`
  The callback URL in Google Cloud does not exactly match the redirect URI
  BirdVision is sending. Check scheme, host, port, path, and whether
  `auth.redirect_uri` or `GOOGLE_REDIRECT_URI` is set correctly.
- Login succeeds but upload is still blocked
  The signed-in email is not listed under `auth.allowed_emails`.
- No login button appears
  Auth is not fully configured, or BirdVision is running in debug mode.
- Test users cannot sign in
  The OAuth consent screen is still in testing mode and the account was not
  added as a test user.
