from pydantic import BaseModel, Field, ValidationError

from api_types import AssistantTurnRequest, AssistantTurnResponse
from services.records import AssistantRunRecord
from state.app_settings import AssistantProviderName

# The shape of stored assistant runs and queued messages. Rows in any other shape load as closed.
ASSISTANT_PAYLOAD_VERSION = 1

_UNREADABLE = "This message can't be loaded."


class AssistantRun(BaseModel):
    request: AssistantTurnRequest
    history: str = "[]"
    provider: AssistantProviderName | None = None
    model: str = ""
    pending_calls: dict[str, str] = Field(default_factory=dict)
    asset_ids: list[str] = Field(default_factory=list)
    # Stored in assistant_jobs, not the payload. Job number to generation id, for the jobs this run queued.
    jobs: dict[str, str] = Field(default_factory=dict, exclude=True)
    # Stored in assistant_jobs. Jobs whose cards this run shows: its own, less any a later run retried, plus any it retried.
    shown: list[str] = Field(default_factory=list, exclude=True)
    # Jobs whose create failed, so the next save deletes their rows. Saves delete nothing else,
    # so an outdated copy of the run cannot drop jobs.
    released: list[str] = Field(default_factory=list, exclude=True)
    retry_count: int = 0
    # Jobs retried at the user's request in this request, so a job that fails again is not retried in a loop.
    user_retried: list[str] = Field(default_factory=list)
    # The recipe nudge fires at most once per request, and not after a recipe job is queued,
    # so continuations after job updates and plain jobs beside a recipe job are not nudged.
    recipe_nudged: bool = False
    response: AssistantTurnResponse | None = None
    stopped: bool = False

    @classmethod
    def load(cls, record: AssistantRunRecord) -> "AssistantRun":
        """A row in an unknown shape loads as a closed message, so it cannot break its chat."""
        if record.payload_version == ASSISTANT_PAYLOAD_VERSION:
            try:
                run = cls.model_validate_json(record.payload)
            except ValidationError:
                pass
            else:
                run.jobs, run.shown = dict(record.jobs), list(record.shown)
                return run
        request = AssistantTurnRequest.model_construct(
            request_id=record.id, conversation_id=record.conversation_id, text=_UNREADABLE, asset_ids=[], provider=None, model=None, origin="desktop",
        )
        return cls(request=request, response=AssistantTurnResponse(text=_UNREADABLE, run_id=record.id),
                   jobs=dict(record.jobs), shown=list(record.shown))

    def record(self) -> AssistantRunRecord:
        return AssistantRunRecord(
            id=self.request.request_id, conversation_id=self.request.conversation_id, payload=self.model_dump_json(),
            payload_version=ASSISTANT_PAYLOAD_VERSION, status=self.response.status if self.response is not None else None,
            stopped=self.stopped, jobs=dict(self.jobs), shown=tuple(self.shown), released=tuple(self.released),
        )

    def add_job(self, job_number: str, generation_id: str) -> None:
        self.jobs[job_number] = generation_id
        if generation_id in self.released:
            self.released.remove(generation_id)
        if generation_id not in self.shown:
            self.shown.append(generation_id)

    def release_job(self, job_number: str) -> None:
        generation_id = self.jobs.pop(job_number, None)
        if generation_id is None:
            return
        self.released.append(generation_id)
        if generation_id in self.shown:
            self.shown.remove(generation_id)

    def show_retried(self, generation_id: str) -> None:
        """Move the card of a job retried in this request here; the store moves it on save."""
        self.user_retried.append(generation_id)
        if generation_id not in self.shown:
            self.shown.append(generation_id)

    def generation_ids(self) -> list[str]:
        return list(self.shown)
