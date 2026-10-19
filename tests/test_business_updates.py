from __future__ import annotations

import asyncio
import json
import os
import stat

import pytest
from aiogram.methods import GetBusinessConnection, SendMessage

from assistant.claude_cli import ClaudeAuthError, ClaudeLimitError, ClaudeTimeout
from assistant.handlers import STRANGER_START_TEXT
from assistant.service import Outcome

from .conftest import CONN_ID, OTHER_ID, OWNER_ID

posix_only = pytest.mark.skipif(os.name != "posix", reason="POSIX file modes")


# ----- preview mode -------------------------------------------------------------


async def test_preview_mode_shows_reply_and_sends_nothing(make_harness):
    h = make_harness()
    await h.connect()
    h.generator.queue("Yaxshi, o'zing-chi?")

    assert await h.business_message("Salom, qalesan?") == Outcome.PREVIEWED

    assert h.session.sent() == []
    assert len(h.previews) == 1
    preview = h.previews[0]
    assert preview.reply == "Yaxshi, o'zing-chi?"
    assert preview.incoming == "Salom, qalesan?"
    assert preview.language == "uz"
    lines = (h.data_dir / "previews.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["reply"] == "Yaxshi, o'zing-chi?"
    # The unsent reply is not part of the history.
    assert [item["role"] for item in h.history()] == ["other"]


@posix_only
async def test_preview_file_is_private(make_harness):
    h = make_harness()
    await h.connect()
    await h.business_message()
    assert stat.S_IMODE((h.data_dir / "previews.jsonl").stat().st_mode) == 0o600
    assert stat.S_IMODE(h.data_dir.stat().st_mode) == 0o700


async def test_known_connection_in_preview_mode_makes_no_telegram_requests(make_harness):
    h = make_harness()
    await h.connect()
    h.session.requests.clear()
    await h.business_message()
    await h.business_message("Bugun bo'shmisan?")
    assert h.session.requests == []


async def test_unknown_connection_is_looked_up_once_then_cached(make_harness):
    h = make_harness()
    h.set_connection()
    assert await h.business_message() == Outcome.PREVIEWED
    assert await h.business_message("Yana bir savol") == Outcome.PREVIEWED
    assert len(h.session.lookups()) == 1
    assert h.session.sent() == []


async def test_connection_lookup_failure_is_no_connection(make_harness):
    h = make_harness()
    assert await h.business_message() == Outcome.NO_CONNECTION
    assert h.generator.calls == []


async def test_message_without_connection_id(make_harness):
    h = make_harness()
    assert await h.business_message(connection_id=None) == Outcome.NO_CONNECTION
    assert h.generator.calls == []


async def test_foreign_account_is_ignored(make_harness):
    h = make_harness()
    await h.connect(user_id=4242)
    assert await h.business_message() == Outcome.FOREIGN_CONNECTION
    assert h.generator.calls == []
    assert h.session.sent() == []


async def test_disabled_connection_is_ignored(make_harness):
    h = make_harness()
    await h.connect(enabled=False)
    assert await h.business_message() == Outcome.CONNECTION_DISABLED
    assert h.generator.calls == []


async def test_any_chat_gets_a_reply_without_allowlist(make_harness):
    h = make_harness()
    await h.connect()
    for chat in (5001, 5002, 777777777):
        assert await h.business_message("Salom", from_id=chat) == Outcome.PREVIEWED
    assert len(h.previews) == 3


# ----- owner messages and takeover -------------------------------------------------


async def test_owner_message_is_recorded_and_never_answered(make_harness):
    h = make_harness()
    await h.connect()
    assert await h.business_message("Hozir yozaman", from_id=OWNER_ID, chat_id=OTHER_ID) == Outcome.OWNER_MESSAGE
    assert h.generator.calls == []
    assert h.history() == [
        {"role": "owner", "name": "You", "text": "Hozir yozaman", "lang": None, "ts": h.clock()}
    ]


async def test_owner_takeover_silences_chat_until_it_expires(make_harness):
    h = make_harness(OWNER_TAKEOVER_MINUTES=15)
    await h.connect()
    await h.business_message("Men o'zim javob beraman", from_id=OWNER_ID, chat_id=OTHER_ID)

    assert await h.business_message("Qachon kelasan?") == Outcome.OWNER_ACTIVE
    # Another chat is not affected.
    assert await h.business_message("Salom", from_id=3003) == Outcome.PREVIEWED

    h.clock.advance(15 * 60 + 1)
    assert await h.business_message("Hali ham kutyapman") == Outcome.PREVIEWED


async def test_takeover_disabled_with_zero(make_harness):
    h = make_harness(OWNER_TAKEOVER_MINUTES=0)
    await h.connect()
    await h.business_message("ok", from_id=OWNER_ID, chat_id=OTHER_ID)
    assert await h.business_message("Salom") == Outcome.PREVIEWED


# ----- bots and duplicates -------------------------------------------------------


async def test_messages_sent_by_business_bot_are_ignored(make_harness):
    h = make_harness()
    await h.connect()
    assert await h.business_message("auto reply", from_id=OWNER_ID, via_business_bot=True) == Outcome.IGNORED_BOT
    assert await h.business_message("I am a bot", is_bot=True) == Outcome.IGNORED_BOT
    assert h.generator.calls == []
    assert h.history() == []


async def test_duplicate_sequential_and_parallel_are_handled_once(make_harness):
    h = make_harness()
    await h.connect()
    payload = h.business_message_payload("Salom", message_id=55)
    await asyncio.gather(*(h.feed({"business_message": payload}) for _ in range(3)))
    await h.feed({"business_message": payload})
    assert list(h.assistant.outcomes).count(Outcome.PREVIEWED) == 1
    assert list(h.assistant.outcomes).count(Outcome.DUPLICATE) == 3
    assert len(h.generator.calls) == 1


async def test_duplicates_are_remembered_across_restart(make_harness):
    h = make_harness()
    await h.connect()
    assert await h.business_message("Salom", message_id=77) == Outcome.PREVIEWED
    h.restart()
    await h.connect()
    assert await h.business_message("Salom", message_id=77) == Outcome.DUPLICATE
    assert len(h.generator.calls) == 1


async def test_same_message_id_in_other_chat_is_not_a_duplicate(make_harness):
    h = make_harness()
    await h.connect()
    assert await h.business_message("Salom", message_id=9, from_id=6001) == Outcome.PREVIEWED
    assert await h.business_message("Salom", message_id=9, from_id=6002) == Outcome.PREVIEWED


# ----- message kinds ------------------------------------------------------------------


async def test_service_message_is_unsupported(make_harness):
    h = make_harness()
    await h.connect()
    outcome = await h.business_message(None, extra={"new_chat_title": "x"})
    assert outcome == Outcome.UNSUPPORTED
    assert h.generator.calls == []


async def test_sticker_and_voice_are_described(make_harness):
    h = make_harness()
    await h.connect()
    sticker = {
        "file_id": "s1", "file_unique_id": "u1", "type": "regular", "width": 512, "height": 512,
        "is_animated": False, "is_video": False, "emoji": "😀",
    }
    voice = {"file_id": "v1", "file_unique_id": "u2", "duration": 3}
    await h.business_message(None, extra={"sticker": sticker})
    await h.business_message(None, extra={"voice": voice})
    await h.business_message(None, extra={"photo": [{"file_id": "p", "file_unique_id": "p", "width": 1, "height": 1}], "caption": "qara"})
    texts = [item["text"] for item in h.history()]
    assert texts == ["[sticker] 😀", "[voice message]", "[photo] qara"]


# ----- history and language hints ---------------------------------------------------


async def test_history_is_separate_per_connection_and_chat(make_harness):
    h = make_harness()
    await h.connect()
    await h.connect(connection_id="conn-2")
    await h.business_message("Secret plan for chat A", from_id=7001)
    await h.business_message("Hello from chat B", from_id=7002)
    await h.business_message("Hello from conn-2", from_id=7001, connection_id="conn-2")

    prompts = [call["user"] for call in h.generator.calls]
    assert "Secret plan" not in prompts[1]
    assert "Secret plan" not in prompts[2]
    assert "Hello from chat B" not in prompts[2]
    assert len(h.history(7001)) == 1
    assert len(h.history(7001, "conn-2")) == 1


@pytest.mark.parametrize(
    "text, language",
    [
        ("Привет, как дела?", "Russian"),
        ("Hi, are you free tonight?", "English"),
        ("Bugun bo'shmisan?", "Uzbek"),
        ("Салом, яхшимисиз?", "Uzbek"),
    ],
)
async def test_language_hint_in_prompt(make_harness, text, language):
    h = make_harness()
    await h.connect()
    await h.business_message(text)
    assert f"Their latest message is in {language}." in h.generator.calls[-1]["user"]


async def test_emoji_uses_last_language_of_chat(make_harness):
    h = make_harness()
    await h.connect()
    await h.business_message("Привет, как дела?")
    await h.business_message("👍")
    assert "Their latest message is in Russian." in h.generator.calls[-1]["user"]
    assert h.previews[-1].language == "ru"


async def test_profile_is_reread_and_comments_removed(make_harness):
    h = make_harness()
    await h.connect()
    await h.business_message("Salom")
    assert "Test Owner" in h.generator.calls[-1]["system"]
    assert "comment that must not reach Claude" not in h.generator.calls[-1]["system"]
    h.profile_path.write_text("- Ism: Yangi Ism\n", encoding="utf-8")
    await h.business_message("Yana salom")
    assert "Yangi Ism" in h.generator.calls[-1]["system"]


async def test_skip_is_not_shown(make_harness):
    h = make_harness()
    await h.connect()
    h.generator.queue("[SKIP]", "  ")
    assert await h.business_message("ok") == Outcome.SKIPPED
    assert await h.business_message("👍") == Outcome.SKIPPED
    assert h.previews == []


# ----- auto-send ---------------------------------------------------------------------


async def test_auto_send_rechecks_connection_then_sends(make_harness):
    h = make_harness(AUTO_SEND="true")
    await h.connect()
    h.session.requests.clear()
    h.generator.queue("Yaxshi, rahmat!")

    assert await h.business_message("Salom, qalesan?") == Outcome.SENT

    kinds = [type(r) for r in h.session.requests]
    assert kinds == [GetBusinessConnection, SendMessage]
    sent = h.session.sent()[0]
    assert sent.chat_id == OTHER_ID
    assert sent.text == "Yaxshi, rahmat!"
    assert sent.business_connection_id == CONN_ID
    assert h.history()[-1]["role"] == "owner"
    assert h.history()[-1]["text"] == "Yaxshi, rahmat!"


async def test_auto_send_splits_long_replies(make_harness):
    h = make_harness(AUTO_SEND="true")
    await h.connect()
    h.generator.queue("a" * 9000)
    assert await h.business_message("Uzun yoz") == Outcome.SENT
    assert [len(m.text) for m in h.session.sent()] == [4000, 4000, 1000]
    assert all(m.business_connection_id == CONN_ID for m in h.session.sent())


@pytest.mark.parametrize("change", [{"can_reply": False}, {"enabled": False}, {"user_id": 4242}])
async def test_auto_send_refuses_without_fresh_permission(make_harness, change):
    h = make_harness(AUTO_SEND="true")
    await h.connect()
    h.set_connection(**change)  # Telegram's answer changed, but no update arrived

    assert await h.business_message("Salom") == Outcome.PERMISSION_DENIED
    assert h.session.sent() == []
    assert h.previews[-1].note.startswith("not sent:")


async def test_auto_send_does_not_send_if_owner_wrote_meanwhile(make_harness):
    h = make_harness(AUTO_SEND="true")
    await h.connect()

    async def owner_writes() -> None:
        h.generator.hook = None
        await h.business_message("Men o'zim yozaman", from_id=OWNER_ID, chat_id=OTHER_ID)

    h.generator.hook = owner_writes
    payload = h.business_message_payload("Qayerdasan?")
    await h.feed({"business_message": payload})

    assert Outcome.OWNER_ACTIVE in h.assistant.outcomes
    assert h.session.sent() == []
    assert "you wrote in this chat" in h.previews[-1].note


# ----- owner commands and /start ----------------------------------------------------


async def test_pause_and_resume_only_for_owner_and_persist(make_harness):
    h = make_harness()
    await h.connect()

    await h.direct_message("/pause", from_id=OTHER_ID)
    assert h.assistant.state.paused is False
    assert h.session.sent() == []

    await h.direct_message("/pause")
    assert h.assistant.state.paused is True
    assert await h.business_message("Salom") == Outcome.PAUSED

    h.restart()
    await h.connect()
    assert h.assistant.state.paused is True
    assert await h.business_message("Salom yana") == Outcome.PAUSED

    await h.direct_message("/resume", from_id=OTHER_ID)
    assert h.assistant.state.paused is True
    await h.direct_message("/resume")
    assert h.assistant.state.paused is False
    assert await h.business_message("Endi-chi?") == Outcome.PREVIEWED


async def test_strangers_get_nothing_for_commands_and_messages(make_harness):
    h = make_harness()
    for text in ("/pause", "/status", "/resume", "hello bot", "What does your owner do?"):
        await h.direct_message(text, from_id=OTHER_ID)
    assert h.session.sent() == []
    assert h.generator.calls == []


async def test_status_for_owner(make_harness):
    h = make_harness()
    await h.direct_message("/status")
    text = h.session.sent()[-1].text
    assert "PREVIEW" in text and "replies are only shown in the logs" in text
    assert "running" in text


async def test_start_for_owner_is_uzbek_guide(make_harness):
    h = make_harness()
    await h.direct_message("/start")
    text = h.session.sent()[-1].text
    assert h.session.sent()[-1].chat_id == OWNER_ID
    for fragment in ("PREVIEW", "ishlayapti", "Telegram Business", "Chatbots", "/pause", "/resume", "/status",
                     "Telegram Premium", "15 daqiqa", "profile.md", "AUTO_SEND=true"):
        assert fragment in text


async def test_start_shows_auto_send_and_paused(make_harness):
    h = make_harness(AUTO_SEND="true")
    h.assistant.pause()
    await h.direct_message("/start")
    text = h.session.sent()[-1].text
    assert "AUTO-SEND" in text
    assert "pauzada" in text
    assert "AUTO_SEND=true qiling" not in text


async def test_start_for_stranger_is_short_and_private(make_harness):
    h = make_harness()
    await h.direct_message("/start", from_id=OTHER_ID)
    sent = h.session.sent()
    assert len(sent) == 1
    assert sent[0].text == STRANGER_START_TEXT
    assert "Test Owner" not in sent[0].text
    assert h.generator.calls == []


# ----- Claude limits and failures ----------------------------------------------------


async def test_limit_waits_until_reset_then_continues(make_harness):
    h = make_harness()
    await h.connect()
    reset = h.clock() + 3600
    h.generator.queue(ClaudeLimitError("limit", resets_at=reset))

    assert await h.business_message("Salom") == Outcome.LIMITED
    assert h.assistant.state.limit_until == reset
    assert await h.business_message("Javob ber") == Outcome.LIMITED
    assert len(h.generator.calls) == 1  # no retry while limited

    h.restart()
    await h.connect()
    assert await h.business_message("Restartdan keyin") == Outcome.LIMITED
    assert len(h.generator.calls) == 1

    h.clock.advance(3601)
    assert await h.business_message("Endi-chi?") == Outcome.PREVIEWED


async def test_limit_without_reset_uses_cooldown(make_harness):
    h = make_harness(LIMIT_COOLDOWN_MINUTES=30)
    await h.connect()
    h.generator.queue(ClaudeLimitError("limit"))
    assert await h.business_message("Salom") == Outcome.LIMITED
    assert h.assistant.state.limit_until == h.clock() + 30 * 60
    h.clock.advance(29 * 60)
    assert await h.business_message("Salom 2") == Outcome.LIMITED
    h.clock.advance(61)
    assert await h.business_message("Salom 3") == Outcome.PREVIEWED


async def test_limit_in_auto_send_notifies_owner_and_resume_clears(make_harness):
    h = make_harness(AUTO_SEND="true")
    await h.connect()
    h.generator.queue(ClaudeLimitError("limit", resets_at=h.clock() + 3600))
    assert await h.business_message("Salom") == Outcome.LIMITED
    sent = h.session.sent()
    assert len(sent) == 1
    assert sent[0].chat_id == OWNER_ID
    assert sent[0].business_connection_id is None
    assert "usage limit" in sent[0].text

    await h.direct_message("/resume")
    assert h.assistant.state.limit_until is None
    h.generator.queue("Javob")
    assert await h.business_message("Endi") == Outcome.SENT


async def test_limit_in_preview_mode_sends_nothing(make_harness):
    h = make_harness()
    await h.connect()
    h.generator.queue(ClaudeLimitError("limit"))
    assert await h.business_message("Salom") == Outcome.LIMITED
    assert h.session.sent() == []


@pytest.mark.parametrize("error", [ClaudeTimeout("too slow"), ClaudeAuthError("not logged in", "detail")])
async def test_errors_send_nothing_to_chat_and_alert_owner_once(make_harness, error):
    h = make_harness(AUTO_SEND="true")
    await h.connect()
    h.generator.queue(error, error)
    assert await h.business_message("Salom") == Outcome.ERROR
    assert await h.business_message("Salom 2") == Outcome.ERROR
    sent = h.session.sent()
    assert all(m.chat_id == OWNER_ID for m in sent)
    assert len(sent) == 1  # not repeated within 10 minutes

    h.clock.advance(601)
    h.generator.queue(error)
    assert await h.business_message("Salom 3") == Outcome.ERROR
    assert len(h.session.sent()) == 2


async def test_errors_in_preview_mode_send_nothing(make_harness):
    h = make_harness()
    await h.connect()
    h.generator.queue(ClaudeTimeout("too slow"))
    assert await h.business_message("Salom") == Outcome.ERROR
    assert h.session.sent() == []
