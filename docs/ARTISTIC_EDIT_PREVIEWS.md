# Compare Artistic Edit previews

Use this when an edit almost works and you want to try another version while keeping the first. For example, a second seed might produce a more legible title or a better-fitting speech bubble.

1. Generate a preview in **Artistic Edit**. Click **Generate alternative** to keep the first version and request a second one with a different seed. Both use the same captured image, edit area, prompt, model, and LoRA settings.
2. Choose a version under **Candidate**. It appears on the right. Use **Compare with** to show the original artwork or the other candidate on the left.
3. For an edit made with **Use selected translation**, you can run **Check lettering** on each version. Completed readings stay with their candidate as you switch between them. [Check generated lettering](LETTERING_CHECK.md).
4. Click **Apply** to put the version shown on the right onto the page. It creates one undoable edit. Choose **Discard** to close both previews.

**Regenerate Same Seed** and **Regenerate New Seed** replace the selected version when the new generation succeeds. The other version stays available. If a request fails or you cancel it, you can return to the completed previews while the same page and source artwork are still current.

Generation runs one request at a time. Comparing results or switching between them uses the existing images. With the Black Forest Labs backend, generating each additional version requires the usual confirmation for a paid request. Previews are temporary. Saving the project after applying a version saves the chosen edit.
