# High-Level Design (HLD): Developer Configuration - API Check Bypass

## 1. Overview

This document outlines the changes made for **CR002: Developer Configuration** to streamline the local development experience. The primary goal of this iteration is to allow developers to easily bypass the mandatory "Connect API Keys" gate that blocks access to the application UI on machines lacking sufficient hardware or proper setup.

The solution ensures that developers can access and work on the application UI (e.g., for integrating ComfyUI workflows) without needing actual API keys or a production-ready local GPU environment, while strictly maintaining the integrity of the production codebase.

## 2. Core Components

### 2.1. Backend Runtime Policy

The single source of truth for whether the application requires API keys is the `decide_force_api_generations` function located in `backend/runtime_config/runtime_policy.py`.

*   **Change:** An environment variable check (`LTX_BYPASS_API_CHECK`) was introduced into this function.
*   **Logic:** If `os.environ.get("LTX_BYPASS_API_CHECK") == "1"`, the function immediately returns `False`.
*   **Impact:** Returning `False` forces the backend to report that local generation is allowed, which in turn signals the frontend to dismiss the blocking API Key modal.

### 2.2. Development Scripts configuration

To ensure this bypass is seamless and strictly limited to local development, the environment variable is injected via `package.json` scripts rather than relying on a global `.env` file.

*   **Change:** The `dev` and `dev:debug` scripts in `package.json` were updated using `cross-env`.
*   **Old Scripts:**
    ```json
    "dev": "vite",
    "dev:debug": "cross-env BACKEND_DEBUG=1 ELECTRON_DEBUG=1 vite"
    ```
*   **New Scripts:**
    ```json
    "dev": "cross-env LTX_BYPASS_API_CHECK=1 vite",
    "dev:debug": "cross-env BACKEND_DEBUG=1 ELECTRON_DEBUG=1 LTX_BYPASS_API_CHECK=1 vite"
    ```

## 3. Security and Production Safety

*   **No Global `.env`:** By avoiding a `.env` file, we prevent accidental commits or persistent global state that could inadvertently disable the API check outside of an active `pnpm dev` session.
*   **Production Build Isolation:** The `build`, `build:skip-python`, and `build:fast` scripts in `package.json` do **not** include the `LTX_BYPASS_API_CHECK` flag. Consequently, packaged release binaries remain secure and will correctly enforce hardware checks and API key requirements for end-users.

## 4. Conclusion

This non-invasive approach provides a frictionless path for frontend and integration development without compromising the strict hardware and licensing gates required for the application's production release.