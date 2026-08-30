/*
 * Progressive enhancement, and nothing else.
 *
 * Every page is complete before this file runs. The lesson body, the lab, the
 * quiz, the kernel context and the call graph's list form are all rendered by
 * the server, which is why the <script> is deferred and why nothing below
 * creates content that was not already there in some form. Block this file
 * and the course is still readable; you just have to reload after saving.
 *
 * Two constraints hold throughout.
 *
 * Nothing here turns data into code. There is no eval, no Function
 * constructor, no assignment to innerHTML. Every value this file handles came
 * from a lesson, a note or the symbol index, and all of it reaches the
 * document through textContent, so a note or a symbol name cannot become an
 * element. That is also what lets the content security policy refuse inline
 * script outright: there is nothing inline to allow.
 *
 * Nothing here is asked where to send a request. URLs come from the server's
 * own markup, and they are checked against the prefixes below before being
 * fetched -- a script that fetches whatever a data attribute says is a script
 * that injected markup can aim anywhere.
 */

(function () {
  "use strict";

  /* The dashboard projection: totals, per-module progress, next lesson. */
  var PROGRESS_URL = "/api/progress";

  /* Combined lessons-and-symbols search, the same endpoint the /search page
     renders from on the server. */
  var SEARCH_URL = "/api/search";

  /* Every mutation and every graph read is addressed under one of these. A
     URL from the page that does not start with one is not ours to fetch. */
  var LESSON_API_PREFIX = "/api/lessons/";
  var SYMBOL_API_PREFIX = "/api/symbols/";

  var SEARCH_DEBOUNCE_MS = 250;

  // ------------------------------------------------------------------
  // Reporting failures
  // ------------------------------------------------------------------

  /* A learning tool that silently swallows a failed save loses notes, so
     every failure lands in the [data-error] region nearest the action, and
     falls back to the page-level one. Both are server-rendered live regions,
     so the message is announced as well as shown. */
  function errorHost(element) {
    var local = element ? element.querySelector("[data-error]") : null;
    return local || document.querySelector("[data-error]");
  }

  function report(host, message) {
    if (host) {
      host.textContent = message;
    }
  }

  function clear(host) {
    if (host) {
      host.textContent = "";
    }
  }

  // ------------------------------------------------------------------
  // Talking to the API
  // ------------------------------------------------------------------

  function localUrl(url, prefixes) {
    if (!url) {
      return null;
    }
    for (var index = 0; index < prefixes.length; index += 1) {
      if (url.indexOf(prefixes[index]) === 0) {
        return url;
      }
    }
    return null;
  }

  /* Errors arrive in one envelope -- {"error": {"code", "message"}} -- so the
     message shown to a learner is the one the server wrote, not "something
     went wrong". */
  async function requestJson(url, options) {
    var response = await fetch(url, options);
    var payload = null;
    try {
      payload = await response.json();
    } catch (error) {
      payload = null;
    }

    if (!response.ok) {
      throw new Error(
        payload && payload.error && payload.error.message
          ? payload.error.message
          : "The request failed."
      );
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

  // ------------------------------------------------------------------
  // Building elements
  // ------------------------------------------------------------------

  function make(tag, className, text) {
    var node = document.createElement(tag);
    if (className) {
      node.className = className;
    }
    if (text !== undefined && text !== null) {
      node.textContent = String(text);
    }
    return node;
  }

  function link(href, text) {
    var anchor = make("a", null, text);
    anchor.setAttribute("href", href);
    return anchor;
  }

  function replaceChildren(host, children) {
    while (host.firstChild) {
      host.removeChild(host.firstChild);
    }
    children.forEach(function (child) {
      host.appendChild(child);
    });
  }

  // ------------------------------------------------------------------
  // Notes
  // ------------------------------------------------------------------

  function bindNoteForm(form) {
    var host = errorHost(form);
    var url = localUrl(form.getAttribute("action"), [
      LESSON_API_PREFIX,
      SYMBOL_API_PREFIX,
    ]);
    var saved = form.querySelector("[data-note-saved]");

    form.addEventListener("submit", async function (event) {
      event.preventDefault();
      clear(host);
      if (!url) {
        report(host, "This note form has no usable address.");
        return;
      }
      var field = form.querySelector("[name=body]");
      try {
        await requestJson(url, jsonRequest("PUT", { body: field.value }));
        if (saved) {
          saved.textContent = "Saved.";
        }
      } catch (error) {
        report(host, error.message);
      }
    });
  }

  // ------------------------------------------------------------------
  // Progress
  // ------------------------------------------------------------------

  /* The buttons post a transition; the course total then comes back from the
     dashboard projection, so the bar a learner watches move is the number the
     server actually holds rather than one this file guessed. Pages without a
     bar -- a lesson page -- skip the request entirely. */
  async function refreshProgress() {
    var bar = document.querySelector('[data-progress="overall"]');
    if (!bar) {
      return;
    }
    var payload = await requestJson(PROGRESS_URL, { method: "GET" });
    if (payload) {
      bar.setAttribute("value", String(payload.percent_complete));
      bar.setAttribute("aria-valuenow", String(payload.percent_complete));
    }
  }

  function bindProgressButton(button) {
    var host = errorHost(button.closest("section"));
    var url = localUrl(button.dataset.progressUrl, [LESSON_API_PREFIX]);

    button.addEventListener("click", async function () {
      clear(host);
      if (!url) {
        report(host, "This control has no usable address.");
        return;
      }
      try {
        var payload = await requestJson(
          url,
          jsonRequest("POST", { status: button.dataset.progressAction })
        );
        var article = button.closest("[data-lesson-id]");
        if (article && payload) {
          article.setAttribute("data-progress-status", payload.status);
        }
        await refreshProgress();
      } catch (error) {
        report(host, error.message);
      }
    });
  }

  // ------------------------------------------------------------------
  // Quiz
  // ------------------------------------------------------------------

  /* The page never knew the answers; the graded response is the first and
     only place an explanation exists, which is why the results block is empty
     in the markup and filled here. */
  function renderQuizResults(host, payload) {
    var children = [
      make(
        "p",
        null,
        "Scored " +
          Math.round(payload.score * 100) +
          "% (" +
          payload.correct_count +
          " of " +
          payload.question_count +
          "). Best so far: " +
          Math.round(payload.best_score * 100) +
          "%."
      ),
    ];

    var list = make("ul");
    payload.results.forEach(function (result) {
      var item = make("li", null, result.correct ? "Correct. " : "Not right. ");
      item.setAttribute("data-correct", result.correct ? "true" : "false");
      if (result.explanation) {
        item.appendChild(make("span", "muted", result.explanation));
      }
      list.appendChild(item);
    });
    children.push(list);

    if (payload.meets_mastery_gate) {
      children.push(make("p", null, "That clears the mastery gate."));
    }
    replaceChildren(host, children);
  }

  function bindQuizForm(form) {
    var host = errorHost(form);
    var results = form.querySelector("[data-quiz-results]");
    var url = localUrl(form.getAttribute("action"), [LESSON_API_PREFIX]);

    form.addEventListener("submit", async function (event) {
      event.preventDefault();
      clear(host);
      if (!url) {
        report(host, "This quiz has no usable address.");
        return;
      }

      var responses = {};
      form.querySelectorAll("[data-quiz-question]").forEach(function (group) {
        var chosen = group.querySelector("input:checked");
        if (chosen) {
          responses[group.dataset.quizQuestion] = chosen.value;
        }
      });

      try {
        var payload = await requestJson(
          url,
          jsonRequest("POST", { responses: responses })
        );
        if (results && payload) {
          renderQuizResults(results, payload);
        }
      } catch (error) {
        report(host, error.message);
      }
    });
  }

  // ------------------------------------------------------------------
  // Reviews
  // ------------------------------------------------------------------

  function bindReviewButton(button) {
    var section = button.closest("[data-review]");
    var host = errorHost(section);
    var url = localUrl(button.dataset.reviewUrl, [LESSON_API_PREFIX]);

    button.addEventListener("click", async function () {
      clear(host);
      if (!url) {
        report(host, "This control has no usable address.");
        return;
      }
      try {
        var card = await requestJson(
          url,
          jsonRequest("POST", {
            correct: button.dataset.reviewAction === "correct",
          })
        );
        if (section && card) {
          section.setAttribute("data-review", String(card.level));
          var state = section.querySelector("[data-review-state]");
          if (state) {
            state.textContent =
              "Box " + card.level + " of " + card.max_level + ".";
          }
        }
      } catch (error) {
        report(host, error.message);
      }
    });
  }

  // ------------------------------------------------------------------
  // Search
  // ------------------------------------------------------------------

  function lessonResultItem(hit) {
    var item = make("li");
    item.appendChild(link(hit.url, hit.title));
    item.appendChild(make("span", "muted", " in " + hit.module_title));
    return item;
  }

  function symbolResultItem(symbol) {
    var item = make("li");
    var code = make("code", null, symbol.name);
    var anchor = link(symbol.url, null);
    anchor.appendChild(code);
    item.appendChild(anchor);
    item.appendChild(
      make("span", "muted", " " + symbol.kind + " in " + symbol.relative_path)
    );
    return item;
  }

  function fillResults(section, items, build, emptyMessage) {
    if (!section) {
      return;
    }
    section.setAttribute("data-count", String(items.length));
    if (!items.length) {
      replaceChildren(section, [make("p", null, emptyMessage)]);
      return;
    }
    var list = make("ul", "results");
    items.forEach(function (entry) {
      list.appendChild(build(entry));
    });
    replaceChildren(section, [list]);
  }

  /* On the /search page the form is taken over so results arrive without a
     reload. Everywhere else it stays an ordinary GET form: submitting it
     navigates, which is what makes a search bookmarkable. */
  function bindSearchForm(form) {
    var lessons = document.querySelector("[data-lesson-results]");
    var symbols = document.querySelector("[data-symbol-results]");
    if (!lessons || !symbols) {
      return;
    }

    var field = form.querySelector("[name=q]");
    var host = errorHost(null);
    var timer = null;

    async function run() {
      var query = field.value.trim();
      if (!query) {
        return;
      }
      clear(host);
      try {
        var payload = await requestJson(
          SEARCH_URL + "?q=" + encodeURIComponent(query),
          { method: "GET" }
        );
        fillResults(
          lessons,
          payload.lessons,
          lessonResultItem,
          "No matches in the course."
        );
        fillResults(
          symbols,
          payload.symbols,
          symbolResultItem,
          payload.symbols_unavailable_reason || "No matches in the kernel index."
        );
        window.history.replaceState(
          null,
          "",
          "/search?q=" + encodeURIComponent(query)
        );
      } catch (error) {
        report(host, error.message);
      }
    }

    form.addEventListener("submit", function (event) {
      event.preventDefault();
      run();
    });
    field.addEventListener("input", function () {
      window.clearTimeout(timer);
      timer = window.setTimeout(run, SEARCH_DEBOUNCE_MS);
    });
  }

  // ------------------------------------------------------------------
  // Call graph
  // ------------------------------------------------------------------

  /* The picture, drawn beside the list the server already rendered. The
     container is aria-hidden: the list is the accessible form, and announcing
     both would say everything twice. Confidence is carried as an attribute so
     the stylesheet -- not this file -- decides how a heuristic edge looks. */
  function graphColumn(heading, edges, emptyMessage) {
    var column = make("div", "graph-column");
    column.appendChild(make("h4", null, heading));
    if (!edges.length) {
      column.appendChild(make("p", "muted", emptyMessage));
      return column;
    }
    edges.forEach(function (edge) {
      var node = make("p", "graph-node", edge.name);
      node.setAttribute("data-provenance", edge.provenance);
      node.setAttribute("data-confidence", edge.confidence);
      column.appendChild(node);
    });
    return column;
  }

  async function drawGraph(container) {
    var url = localUrl(container.dataset.graphUrl, [SYMBOL_API_PREFIX]);
    if (!url) {
      return;
    }
    var host = errorHost(null);
    try {
      var payload = await requestJson(url, { method: "GET" });
      replaceChildren(container, [
        graphColumn("Incoming", payload.incoming, "No known callers."),
        graphColumn("This symbol", [
          {
            name: payload.symbol.name,
            provenance: "definition",
            confidence: "high",
          },
        ]),
        graphColumn("Outgoing", payload.outgoing, "No known callees."),
      ]);
    } catch (error) {
      /* The list fallback is already on the page, so a failed draw is a
         missing picture rather than missing information. */
      report(host, error.message);
    }
  }

  // ------------------------------------------------------------------
  // Wiring
  // ------------------------------------------------------------------

  function ready() {
    document.querySelectorAll("[data-note-form]").forEach(bindNoteForm);
    document
      .querySelectorAll("[data-progress-action]")
      .forEach(bindProgressButton);
    document.querySelectorAll("[data-quiz-form]").forEach(bindQuizForm);
    document
      .querySelectorAll("[data-review-action]")
      .forEach(bindReviewButton);
    document.querySelectorAll("[data-search-form]").forEach(bindSearchForm);
    document.querySelectorAll("[data-symbol-graph]").forEach(drawGraph);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", ready);
  } else {
    ready();
  }
})();
