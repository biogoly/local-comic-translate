# Project glossary and translation context

Open **Translation context** below **Fit Text** in the editing sidebar. These settings belong to the current comic project and are saved with it, including recovery saves. They are separate from the general **LLM context** setting, which still supplies your usual translation preferences.

## Glossary

Add names, recurring terms, and their preferred translations. **Add selected source/translation** starts an entry using the selected text region. You can edit the term before saving it.

Each entry supports aliases, a language pair, a note, a matching rule, and Enabled/Approved switches. Only enabled, approved entries that match the request's source text and language pair enter the prompt. A blank language scope applies to any language; Portuguese and Brazilian Portuguese remain distinct scopes.

Automatic matching normalizes Unicode and letter case. It uses whole-word matching for languages with word separators and substring matching for Japanese, Chinese, and Thai. Choose Whole word or Substring explicitly when a name or expression needs different behavior. Matching guides the model through its prompt; it does not replace substrings in completed translations.

For example, set French `Capitaine Étoile` to English `Captain Star`, with `Étoile` as an alias. The term is included for regions containing either form. Changing an unrelated entry does not invalidate cached requests that never included it.

### Suggest glossary terms from reviewed dialogue

Set the comic’s source and target languages and approve some translations using **Review → Translation → Mark reviewed**. Open **Translation context → Glossary → Suggest glossary terms…**. The dialog shows the model destination and the approved dialogue pairs it will use. Click **Generate** to ask the selected LLM for candidate names and terms. You can use suggestions even when earlier-page translation history is switched off.

Inspect each candidate alongside its source and translation. For example, if an approved passage calls *Linterna Roja* the *Vermilion Lantern*, you can keep that as the place’s name for the rest of the comic. Edit the proposed wording as needed, select the terms you want and add them to the glossary draft. Click **Apply** to make those entries available to future translations. **Undo context** reverses that project edit.

The app checks that both sides of a model suggestion occur in the same approved example and checks the source term against existing terms and aliases in the applicable languages. Read the example to decide whether the expressions mean the same thing and are useful beyond that passage.

## Characters and notes

Add a character's name, aliases, preferred translated name, and voice/formality notes in **Characters**. Select a text region in the page and assign its speaker in the editor. Character notes apply only to assigned regions; the app does not infer speakers automatically. Assignments use persistent page and region IDs and survive saving, page reordering, and matching regions during detection.

Use **Notes** for project-wide directions and a story summary. Both are included in eligible LLM translation requests. You can write the summary yourself or review a model suggestion.

## Use earlier approved translations

1. On a translated page, select a region. In **Review**, choose **Translation** and click **Mark reviewed** once the wording is right.
2. In **Translation context → Notes**, enable **Include approved translations from earlier pages** and click **Apply**. The option starts off in new and older projects.
3. On a later page, open **Request preview → Build preview** to inspect the examples the translator will receive.

History follows the project’s current page order and the explicit source and target languages. For a request spanning several pages, examples come from before the earliest page in that request. Choose a specific source language when working with history. An **Auto** script classification such as Latin can correspond to several languages. European and Brazilian Portuguese use separate histories.

The app selects up to 24 recent, distinct source/translation pairs that fit after reserving space for the response, glossary, character notes and other required context. If you change an approved source or translation, review the new wording to make it eligible again. A **Keep translation** protection and a translation review serve different purposes. Mark the translation reviewed to use it as an example.

## Suggest a story summary

1. In **Translation context → Notes**, click **Suggest story summary…**. This also works with the history checkbox off.
2. Inspect the current summary, the chosen model destination and the approved examples. Summary suggestions can draw on reviewed translations anywhere in this project for the selected language pair, including pages after the current one. **Generate** sends the displayed text to the selected LLM.
3. Read and edit the proposed summary, then click **Use suggestion** to copy it into the Notes editor. Click **Apply** to save it to the project. **Undo context** restores the previous applied summary.

The model writes the suggestion in the target language. Check names, relationships and plot details before applying it. Closing the suggestion dialog discards the proposal. If the project, selected examples or model destination changes during generation, open a fresh suggestion.

## Apply, undo, and preview

- **Apply** saves the current draft to the project and marks the project changed. Subsequent translations use it; existing lettering is not retranslated automatically.
- **Undo context / Redo context** affect project context only. Page-level image, inpainting, and lettering undo remain separate.
- **Revert draft** discards unapplied editor changes. Closing and reopening the editor retains a draft within the same project; opening another project clears it.
- **Build preview** shows the request for the selected region or current page using the editor's current draft. It includes the applicable context, instructions, sanitized provider/model information, image-inclusion status, and a token estimate. It does not start a server, download a model, or send a translation request.
- The page preview excludes regions that normal page translation preserves. The selected-region preview represents an explicit selected translation. In webtoon mode, Current page previews the physical page; a visible-area or automatic chunk request can contain different regions and image context.

The context budget uses a conservative UTF-8 estimate rather than the model's tokenizer. Output capacity is reserved first when the configured model window is known. If required context does not fit, translation reports the conflict instead of silently omitting terms. Increase the relevant budget/window or shorten the notes. Image token counts are projector-dependent; the preview includes an estimated allowance.

## Backends and existing work

Local LLM and the supported LLM provider engines use the same context/request builder. Traditional translation services keep their existing behavior; structured prompt preview is unavailable for those services. API keys and local model paths are not stored in the project profile or displayed in request previews.

Selected-region, current-page, selected-pages, automatic, and webtoon translation use frozen request inputs. Cache identity includes the provider/model, nonsecret endpoint, languages, generation settings, relevant context, source regions, and the image when sent. Raw model results are cached before uppercase/layout processing; manually edited or protected targets are not used as fresh model results.

Results are matched back by region ID. Missing, duplicate, unknown, or invalid response entries reject the entire request before replacing targets. Changes to source/target text, protections, project context, project identity, or cancellation can invalidate a pending result. Previously completed independent pages remain intact. An external server that silently changes weights behind an unchanged model name cannot be detected without new server information.

Existing manual targets and **Keep translation** regions retain their normal protections. To intentionally translate an existing selected region again, use its Translate action; protected regions retain the review/preview flow. Changing the glossary alone never overwrites them.

## Suggested first test

1. Open a project with a recurring character name and add one language-scoped glossary entry.
2. Build the current-page preview and confirm that matching regions include the preferred name.
3. Apply and translate a region with that name using your usual local model or provider.
4. Change the preferred name, apply, and explicitly translate that region again to verify it uses the new context rather than an old cached answer.
5. Save/reopen the project, then open a different project and check that its glossary and notes are independent.

Validation on November 3, 2026: all 727 automated tests and 18 source smoke checks passed. These cover Unicode/dialect matching, persistence/recovery, request identity and response validation, staged jobs, stale-result handling, UI undo/drafts, and two-chunk webtoon rendering/finalization. All nine interface catalogs passed source/compiled-resource checks; the editor was checked across all six themes. A disposable Windows GUI check verified the editor and local request preview with network/model access disabled. Translation quality and adherence to terminology still require a real model and representative comics. No package was rebuilt for this milestone.
