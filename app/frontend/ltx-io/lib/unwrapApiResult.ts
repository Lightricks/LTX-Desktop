/** Error thrown by `unwrapApiResult`, retaining the API error `code` and status. */
export class ApiResultError extends Error {
  readonly code?: string;
  readonly status?: unknown;

  constructor(
    message: string,
    options?: { code?: string; status?: unknown },
  ) {
    super(message);
    this.name = "ApiResultError";
    this.code = options?.code;
    this.status = options?.status;
  }
}

/** Throw when an `ApiClient` result is `{ ok: false }`, otherwise return `data`. */
export type ApiResult<T> =
  | { ok: true; data: T }
  | { ok: false; status?: unknown; error: { message: string; code?: string } };

export function unwrapApiResult<T>(result: ApiResult<T>): T {
  if (!result.ok) {
    throw new ApiResultError(result.error.message, {
      code: result.error.code,
      status: result.status,
    });
  }
  return result.data;
}
