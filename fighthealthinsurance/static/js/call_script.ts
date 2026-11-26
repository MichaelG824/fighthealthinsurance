// The phone script page (call_script.html, call_script.py).
//
// Print is the browser's own: the page's print rules keep the script and
// its notes sheet. For a case, the page asks our models to tailor two parts
// of the script ("why I'm calling" and the questions about the reason) and
// swaps them in. The script is complete without them, so when that fails,
// times out, or the models' words fail the server's checks, the template's
// words stay and nothing is said.

const TAILOR_TIMEOUT_MS = 60000;

interface Tailored {
  why_calling: string;
  questions: string[];
}

function isTailored(value: unknown): value is Tailored {
  const candidate = value as Tailored | null;
  return (
    !!candidate &&
    typeof candidate.why_calling === "string" &&
    Array.isArray(candidate.questions) &&
    candidate.questions.length > 0 &&
    candidate.questions.every((question) => typeof question === "string")
  );
}

function quoted(text: string): string {
  return "“" + text + "”";
}

// The model's words go in as text, never as markup.
function tailoredTag(): HTMLSpanElement {
  const tag = document.createElement("span");
  tag.className = "fhi-call-tailored-tag";
  tag.textContent = "Tailored to your denial";
  return tag;
}

function applyTailoring(tailored: Tailored): boolean {
  const why = document.getElementById("call-script-why");
  const list = document.getElementById("call-script-questions");
  if (!why || !list) {
    return false;
  }
  const slots = Array.from(list.querySelectorAll("li[data-call-tailorable]"));
  if (slots.length === 0) {
    return false;
  }
  why.textContent = quoted(tailored.why_calling);
  why.append(" ", tailoredTag());
  for (const question of tailored.questions) {
    const item = document.createElement("li");
    item.setAttribute("data-call-tailorable", "");
    item.textContent = quoted(question);
    item.append(" ", tailoredTag());
    list.insertBefore(item, slots[0]);
  }
  slots.forEach((slot) => slot.remove());
  return true;
}

async function tailorScript(form: HTMLFormElement, status: HTMLElement): Promise<void> {
  status.textContent =
    "Tailoring two parts of this script to your denial. You can start reading: it takes up to a minute.";
  status.hidden = false;
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), TAILOR_TIMEOUT_MS);
  try {
    const response = await fetch(form.action, {
      method: "POST",
      body: new FormData(form),
      credentials: "same-origin",
      signal: controller.signal,
    });
    const data = response.ok ? await response.json() : null;
    if (data && isTailored(data.tailored) && applyTailoring(data.tailored)) {
      status.textContent = 'Done: the parts marked "Tailored to your denial" were written for your case.';
      return;
    }
  } catch {
    // Timed out or failed: the template's script stays, and it is complete.
  } finally {
    window.clearTimeout(timer);
  }
  status.hidden = true;
}

function setupCallScript(): void {
  const printButton = document.getElementById("call-script-print");
  if (printButton) {
    printButton.addEventListener("click", () => window.print());
  }
  const form = document.getElementById("call-script-tailor") as HTMLFormElement | null;
  const status = document.getElementById("call-script-tailor-status");
  if (form && status) {
    void tailorScript(form, status);
  }
}

setupCallScript();
