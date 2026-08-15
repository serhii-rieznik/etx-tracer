Project Coding Rules

- Do not implement hacks, workarounds, empirical fitting, special-case compensation, or data-matching patches unless the user explicitly asks for that approach and confirms it after the tradeoffs are stated.
- Do not commit one-off validation or development artifacts. This includes temporary test cases added only to validate the current change, ad hoc test programs, test scenes, validation scripts, rendered images, comparison reports, and generated test executables. Keep these local and ignored, and remove temporary changes to tracked test files before committing.
- Maintained, reusable validation infrastructure may be committed when it belongs to the project proper. Shared validation shaders must live in the common `sources/etx/render/shaders/` directory; they must not be placed in a task-specific validation or assets folder.
- Permanent regression tests are appropriate only when they are intended to remain maintained as part of the regular test suite. Do not turn a one-time implementation check into a permanent test without explicit user approval.
