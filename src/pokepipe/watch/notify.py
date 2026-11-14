"""Desktop notifications from the watcher: a review is waiting, posts are ready, a
video failed. So nobody has to sit and watch the terminal.

Windows only (a toast through Windows PowerShell, no extra packages); elsewhere it is
a no-op. Notifying must never break the pipeline, so every failure is swallowed.
"""

from __future__ import annotations

import shutil
import subprocess
import sys

# Toasts need a registered app id; Windows PowerShell's own one always exists.
_APP_ID = "{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\\WindowsPowerShell\\v1.0\\powershell.exe"


def _ps_quote(text: str) -> str:
    """A PowerShell single-quoted string literal ('' escapes ')."""
    return "'" + text.replace("'", "''") + "'"


def toast_script(title: str, message: str) -> str:
    return "; ".join([
        "[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, "
        "ContentType = WindowsRuntime] | Out-Null",
        "$t = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent("
        "[Windows.UI.Notifications.ToastTemplateType]::ToastText02)",
        "$n = $t.GetElementsByTagName('text')",
        f"$n.Item(0).AppendChild($t.CreateTextNode({_ps_quote(title)})) | Out-Null",
        f"$n.Item(1).AppendChild($t.CreateTextNode({_ps_quote(message)})) | Out-Null",
        "[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("
        f"{_ps_quote(_APP_ID)}).Show([Windows.UI.Notifications.ToastNotification]::new($t))",
    ])


def notify(title: str, message: str) -> bool:
    """Show a toast; True if one was sent. Never raises, never blocks the watcher."""
    if sys.platform != "win32":
        return False
    exe = shutil.which("powershell") or shutil.which("powershell.exe")
    if not exe:
        return False
    try:
        subprocess.Popen(
            [exe, "-NoProfile", "-NonInteractive", "-WindowStyle", "Hidden", "-Command",
             toast_script(title, message)],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return True
    except OSError:
        return False
