# Cloud Atlas on this laptop

Double-click **Start Cloud Atlas.cmd** in the project folder. The website opens at
http://localhost:7860 after loading its installed models. Keep the launcher window
open; press Ctrl+C to stop. Internet is needed for Google login and satellite imagery.
No notebook, model upload, ZIP attachment, hosting subscription, or admin password
is needed during use. Dashboard, processing, results history, How It Works, and
About remain in the existing portal.

## One-time account setup

Google requires your own OAuth application credentials even for a local website.
In Google Cloud Console, configure Google Auth Platform, select External audience,
and add your Google account as a test user while the application is in Testing.
Create an OAuth client of type **Web application** with this exact redirect URI:

`http://localhost:7860/api/auth/google/callback`

Open `local_settings.json` in a text editor. Enter the Google client ID and secret,
and your Copernicus satellite API client ID and secret. Keep the redirect URI as
provided. These are owner setup credentials; visitors sign in only with Google.
Do not paste secrets into chat or commit this file. To allow accounts beyond test
users, configure the Google application's production audience as Google requires.

Once configured, future sessions need only the launcher and Google sign-in.
Always use **localhost**, rather than changing the hostname to 127.0.0.1, so the
login callback and browser cookies match.

## Installed data

- `local_models/`: the two retained trained checkpoints, loaded automatically.
- `local_results/`: account-scoped images, history, and the session database.
- `local_settings.json`: private configuration, excluded from Git.

Keep these folders on the laptop. Google and satellite credentials have not been
verified until a real sign-in and imagery request succeed. Sharing a Google account
does not make the localhost website accessible from another person's computer.

Google's setup documentation: https://developers.google.com/identity/protocols/oauth2/web-server
