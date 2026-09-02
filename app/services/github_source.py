"""
DEPRECATED / UNUSED -- the GitHub connector was replaced by the Google
Drive connector (see app/services/googledrive_source.py). Nothing in this
codebase imports this module anymore (app/api/routes/sources.py now
imports googledrive_source instead).

Kept on disk only because this session doesn't have delete access to your
filesystem -- safe to delete this file yourself:
  taaaidocassist_api/app/services/github_source.py

You can also remove PyGithub from requirements.txt if it's still there
(it should already be replaced by google-auth).
"""
