# High-Level Design (HLD): Developer Configuration - API Check Bypass

## 1. Overview

This document outlines the changes made for **CR002: Developer Configuration** to streamline the local development experience. The primary goal of this iteration is to allow developers to easily bypass the mandatory "Connect API Keys" gate that blocks access to the application UI on machines lacking sufficient hardware or proper setup.

The solution ensures that developers can access and work on the application UI (e.g., for integrating ComfyUI workflows) without needing actual API keys or a production-ready local GPU environment, while strictly maintaining the integrity of the production codebase.

## 2. Core Components

### 2.1. Backend Runtime Policy & Model Checking

The application normally enforces two major gates before allowing access to the main UI:
1.  **API Key Gate:** Triggered if the system lacks the necessary hardware to run local models (managed by `decide_force_api_generations`).
2.  **Model Download Gate:** Triggered if local generation is allowed but the required model files (e.g., checkpoints) are missing on disk.

To bypass these gates, two environment variables were introduced:

*   **`LTX_BYPASS_API_CHECK`:**
    *   **Location:** `backend/runtime_config/runtime_policy.py` (`decide_force_api_generations`)
    *   **Logic:** If set to `"1"`, the function returns `False`, tricking the app into believing local generation is fully supported, thereby dismissing the mandatory API key prompt.
*   **`LTX_BYPASS_MODEL_CHECK`:**
    *   **Location:** `backend/ltx2_server.py`
    *   **Logic:** If set to `"1"`, the `REQUIRED_MODEL_TYPES` frozen set is overridden to be empty (`frozenset()`). This tricks the frontend's model status check into believing all necessary models are already downloaded, skipping the "Choose Location" installation screen.

### 2.2. Development Scripts configuration

To ensure these bypasses are seamless and strictly limited to local development, the environment variables are injected via `package.json` scripts rather than relying on a global `.env` file.

*   **Change:** The `dev` and `dev:debug` scripts in `package.json` were updated using `cross-env`.
*   **Old Scripts:**
    ```json
    "dev": "vite",
    "dev:debug": "cross-env BACKEND_DEBUG=1 ELECTRON_DEBUG=1 vite"
    ```
*   **New Scripts:**
    ```json
    "dev": "cross-env LTX_BYPASS_API_CHECK=1 LTX_BYPASS_MODEL_CHECK=1 vite",
    "dev:debug": "cross-env BACKEND_DEBUG=1 ELECTRON_DEBUG=1 LTX_BYPASS_API_CHECK=1 LTX_BYPASS_MODEL_CHECK=1 vite"
    ```

## 3. Security and Production Safety

*   **No Global `.env`:** By avoiding a `.env` file, we prevent accidental commits or persistent global state that could inadvertently disable the API or model checks outside of an active `pnpm dev` session.
*   **Production Build Isolation:** The `build`, `build:skip-python`, and `build:fast` scripts in `package.json` do **not** include these bypass flags. Consequently, packaged release binaries remain secure and will correctly enforce hardware checks, API key requirements, and model downloads for end-users.

## 4. Conclusion

This non-invasive approach provides a frictionless path for frontend and integration development without compromising the strict hardware and licensing gates required for the application's production release.