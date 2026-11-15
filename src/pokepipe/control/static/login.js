"use strict";

document.getElementById("form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const error = document.getElementById("error");
  error.hidden = true;
  const res = await fetch("/api/login", {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Pokepipe": "1" },
    body: JSON.stringify({ code: document.getElementById("code").value }),
  });
  if (res.ok) {
    location.href = "/";
    return;
  }
  const body = await res.json().catch(() => ({}));
  error.textContent = body.detail || "Sign-in failed";
  error.hidden = false;
});
