/*
 * Progressive enhancement only. Every lesson, module and symbol page is
 * fully readable with this file blocked -- it is served from a <script defer>
 * and everything below binds to markup the server already rendered. Nothing
 * here evaluates a string as code, and nothing writes markup: text goes in
 * through textContent so a note or a search result cannot become an element.
 */

(function () {
  "use strict";

  function reportError(host, message) {
    if (!host) {
      return;
    }
    host.textContent = message;
  }

  function clearError(host) {
    if (host) {
      host.textContent = "";
    }
  }

  async function requestJson(url, options) {
    const response = await fetch(url, options);
    let payload = null;
    try {
      payload = await response.json();
    } catch (error) {
      payload = null;
    }

    if (!response.ok) {
      const detail =
        payload && payload.error && payload.error.message
          ? payload.error.message
          : "The request failed.";
      throw new Error(detail);
    }
    return payload;
  }

  function jsonRequest(method, body) {
    return {
      method: method,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    };
  }

  function bindNoteForm(form) {
    const errorHost = form.querySelector("[data-api-error]");
    form.addEventListener("submit", async function (event) {
      event.preventDefault();
      clearError(errorHost);
      const field = form.querySelector("[name=body]");
      try {
        await requestJson(form.action, jsonRequest("PUT", { body: field.value }));
      } catch (error) {
        reportError(errorHost, error.message);
      }
    });
  }

  function bindProgressButton(button) {
    const errorHost = document.querySelector("[data-api-error]");
    button.addEventListener("click", async function () {
      clearError(errorHost);
      try {
        await requestJson(
          button.dataset.progressUrl,
          jsonRequest("POST", { status: button.dataset.progressAction })
        );
        window.location.reload();
      } catch (error) {
        reportError(errorHost, error.message);
      }
    });
  }

  function ready() {
    document.querySelectorAll("[data-note-form]").forEach(bindNoteForm);
    document
      .querySelectorAll("[data-progress-action]")
      .forEach(bindProgressButton);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", ready);
  } else {
    ready();
  }
})();
