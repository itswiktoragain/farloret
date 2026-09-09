"""Farloret compatibility fixes layered over the upstream Open WebUI runtime.

These shims intentionally stay small and reversible. They repair regressions
present in the Scarlet upstream snapshot without copying the enormous middleware
module into the fork.
"""
from __future__ import annotations

import logging
import threading
from functools import wraps

log = logging.getLogger(__name__)
_installed = False
_mps_lock = threading.Lock()


def _legacy_chain(chat, message_id, get_message_list):
    if not chat or not isinstance(getattr(chat, "chat", None), dict):
        return []
    history = chat.chat.get("history") or {}
    messages = history.get("messages") or {}
    if not isinstance(messages, dict) or message_id not in messages:
        return []
    return get_message_list(messages, message_id) or []


def _needs_legacy_fallback(db_messages, legacy_messages) -> bool:
    """Use legacy history only when it contains ancestors missing from DB replay."""
    if not legacy_messages:
        return False
    if not db_messages:
        return True

    db_ids = [item.get("id") for item in db_messages if isinstance(item, dict) and item.get("id")]
    legacy_ids = [item.get("id") for item in legacy_messages if isinstance(item, dict) and item.get("id")]

    # A normalized chain may look internally valid while starting halfway through
    # the visible conversation. The embedded history is the authoritative fallback
    # for those missing ancestors.
    return bool(legacy_ids) and (
        len(legacy_ids) > len(db_ids)
        or (db_ids and legacy_ids[-1] == db_ids[-1] and not set(legacy_ids).issubset(set(db_ids)))
    )


class _LockedLocalModel:
    """Serialize Torch MPS model inference while forwarding every other attribute."""

    def __init__(self, target):
        self._target = target

    def __getattr__(self, name):
        return getattr(self._target, name)

    def encode(self, *args, **kwargs):
        with _mps_lock:
            return self._target.encode(*args, **kwargs)

    def predict(self, *args, **kwargs):
        with _mps_lock:
            return self._target.predict(*args, **kwargs)


def install() -> None:
    global _installed
    if _installed:
        return
    _installed = True

    from open_webui import env
    from open_webui.models.chats import Chats
    from open_webui.utils import context_compaction
    from open_webui.utils import files as file_utils
    from open_webui.utils import middleware
    from open_webui.utils.misc import get_message_list, get_text_content

    # 1) Conversation replay: never let a shorter normalized DB branch amputate
    # ancestors that are still present in the persisted visible chat history.
    original_load_messages = middleware.load_messages_from_db

    @wraps(original_load_messages)
    async def load_messages_from_db(chat_id: str, message_id: str):
        db_messages = await original_load_messages(chat_id, message_id)
        try:
            chat = await Chats.get_chat_by_id(chat_id)
            legacy_messages = _legacy_chain(chat, message_id, get_message_list)
            if _needs_legacy_fallback(db_messages, legacy_messages):
                log.warning(
                    "Farloret repaired incomplete DB chat replay for %s (%s DB messages, %s persisted messages)",
                    chat_id,
                    len(db_messages or []),
                    len(legacy_messages),
                )
                replay_keys = middleware.MESSAGE_REPLAY_KEYS
                return [{key: value for key, value in msg.items() if key in replay_keys} for msg in legacy_messages]
        except Exception:
            log.exception("Farloret history fallback check failed; keeping normalized DB replay")
        return db_messages

    middleware.load_messages_from_db = load_messages_from_db

    # 2) Tool image regression: persist data-URI images immediately. This keeps
    # multi-megabyte base64 strings out of replay/output structures and leaves a
    # normal authenticated file URL for the UI.
    original_process_tool_result = middleware.process_tool_result

    @wraps(original_process_tool_result)
    async def process_tool_result(
        request,
        tool_function_name,
        tool_result,
        tool_type,
        direct_tool=False,
        metadata=None,
        user=None,
    ):
        result, result_files, embeds = await original_process_tool_result(
            request,
            tool_function_name,
            tool_result,
            tool_type,
            direct_tool,
            metadata,
            user,
        )
        for file_item in result_files or []:
            image_url = file_item.get("url") if isinstance(file_item, dict) else None
            if isinstance(file_item, dict) and file_item.get("type") == "image" and isinstance(image_url, str) and image_url.startswith("data:"):
                try:
                    stored_url = await file_utils.get_file_url_from_base64(
                        request,
                        image_url,
                        {
                            "chat_id": (metadata or {}).get("chat_id"),
                            "message_id": (metadata or {}).get("message_id"),
                            "session_id": (metadata or {}).get("session_id"),
                        },
                        user,
                    )
                    if stored_url:
                        file_item["url"] = stored_url
                except Exception:
                    log.exception("Failed to persist tool-result image")
        return result, result_files, embeds

    middleware.process_tool_result = process_tool_result

    # 3) Context estimator regression (#29818): only textual parts contribute to
    # text-token estimation. Image/file payload cost is estimated separately.
    def estimate_content_tokens(content) -> int:
        if not content:
            return 0
        text_content = get_text_content(content)
        if isinstance(text_content, str):
            text = text_content
        elif isinstance(text_content, list):
            text = "\n".join(
                str(item.get("text", "")) if isinstance(item, dict) else str(item)
                for item in text_content
            )
        else:
            text = str(text_content or "")
        return len(text) // 4

    context_compaction.estimate_content_tokens = estimate_content_tokens

    # 4) Apple MPS crash (#29735): serialize shared local model inference only on
    # MPS. CPU/CUDA and external embedding/reranking services keep their concurrency.
    if getattr(env, "DEVICE_TYPE", "") == "mps":
        from open_webui.retrieval import utils as retrieval_utils
        from open_webui.routers import evaluations

        original_get_embedding_function = retrieval_utils.get_embedding_function
        original_get_reranking_function = retrieval_utils.get_reranking_function

        @wraps(original_get_embedding_function)
        def get_embedding_function(
            embedding_engine,
            embedding_model,
            embedding_function,
            url,
            key,
            embedding_batch_size,
            azure_api_version=None,
            enable_async=True,
            concurrent_requests=0,
        ):
            if embedding_engine == "" and embedding_function is not None:
                embedding_function = _LockedLocalModel(embedding_function)
            return original_get_embedding_function(
                embedding_engine,
                embedding_model,
                embedding_function,
                url,
                key,
                embedding_batch_size,
                azure_api_version,
                enable_async,
                concurrent_requests,
            )

        @wraps(original_get_reranking_function)
        def get_reranking_function(reranking_engine, reranking_model, reranking_function, reranking_batch_size=32):
            if reranking_engine != "external" and reranking_function is not None:
                reranking_function = _LockedLocalModel(reranking_function)
            return original_get_reranking_function(
                reranking_engine,
                reranking_model,
                reranking_function,
                reranking_batch_size,
            )

        retrieval_utils.get_embedding_function = get_embedding_function
        retrieval_utils.get_reranking_function = get_reranking_function

        original_compute_similarities = evaluations._compute_similarities

        @wraps(original_compute_similarities)
        def compute_similarities(*args, **kwargs):
            with _mps_lock:
                return original_compute_similarities(*args, **kwargs)

        evaluations._compute_similarities = compute_similarities
