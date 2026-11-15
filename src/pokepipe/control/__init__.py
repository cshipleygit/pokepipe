"""Control room: a phone-friendly page to send videos to the pipeline and watch them run.

Unlike the review page (loopback only: it streams source footage and edits review
state), this server can listen on the home network so a phone can reach it. So it
exposes only what a phone needs, behind an access code: chunked uploads into the
inbox, job status, retrying a failed job, and the rendered posts.
"""
