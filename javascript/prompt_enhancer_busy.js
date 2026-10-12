// Enhance and Remix are locked while Forge generates an image.
//
// Ollama and the image model share the GPU. An enhancer call that starts
// while an image is generating finds the memory taken, Ollama loads its
// model into system RAM instead, and it stays there: every later prompt is
// slow until Forge is restarted. The other direction (Generate locked while
// the enhancer runs) is wired in scripts/prompt_enhancer.py.
//
// "An image is generating" is read from the page itself: Forge shows the
// Interrupt button (or its "Interrupting..." placeholder) for exactly that
// time, also after a page reload that picks a running job up again. The
// lock is a class on the button and nothing else, so gradio's own state of
// the button is never touched.
(function () {
    const TABS = ["txt2img", "img2img"];
    const SIGNS = ["_interrupt", "_interrupting"];
    const LOCKED = ["_pe_enhance_btn", "_pe_refine_btn"];
    const CLASS = "pe-locked";

    function generating() {
        return TABS.some((tab) =>
            SIGNS.some((sign) => {
                const el = gradioApp().getElementById(tab + sign);
                return el !== null && el.style.display === "block";
            }),
        );
    }

    function apply() {
        const busy = generating();
        for (const tab of TABS) {
            for (const name of LOCKED) {
                const el = gradioApp().getElementById(tab + name);
                // Only on a difference: this runs from an observer of the
                // same attribute.
                if (el !== null && el.classList.contains(CLASS) !== busy) el.classList.toggle(CLASS, busy);
            }
        }
    }

    onUiLoaded(function () {
        const observer = new MutationObserver(apply);
        let watched = 0;
        for (const tab of TABS) {
            for (const sign of SIGNS) {
                const el = gradioApp().getElementById(tab + sign);
                if (el === null) continue;
                observer.observe(el, { attributes: true, attributeFilter: ["style"] });
                watched += 1;
            }
            for (const name of LOCKED) {
                // gradio may rewrite a button's class list when it redraws it.
                const el = gradioApp().getElementById(tab + name);
                if (el !== null) observer.observe(el, { attributes: true, attributeFilter: ["class"] });
            }
        }
        if (watched !== TABS.length * SIGNS.length) {
            console.error(`[PromptEnhancer] ${watched} of ${TABS.length * SIGNS.length} Interrupt buttons found: Enhance and Remix are NOT locked while an image generates`);
        }
        apply();
    });
})();
