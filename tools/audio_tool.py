"""
Audio playback and media trigger tool for Voice AI.
Supports local media discovery, client playback events, queued speaker audio,
and graceful AI direct narration fallback when media files are not found.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional
from edgevoiceai.tools.base import BaseTool, ToolResult, ToolExecutionContext


class AudioPlaybackTool(BaseTool):
    """
    A configurable tool that triggers audio media playback.
    """

    def __init__(
        self,
        name: str,
        description: str,
        parameters_schema: Dict[str, Any],
        media_directory: str = "audio_experiences",
        default_audio_url: str = "",
        speaker_name_field: str = "farmer_name",
        default_speaker_name: str = "",
        district_field: str = "district",
        default_district: str = "",
        experience_years_field: str = "experience_years",
        default_experience_years: str = "",
        client_event_type: str = "play_experience_audio",
        success_status: str = "queued",
        not_found_status: str = "audio_not_found",
        success_message_builder: Optional[Callable[[Dict[str, Any]], str]] = None,
        fallback_narrate_builder: Optional[Callable[[Dict[str, Any]], str]] = None,
        prompt_instruction: Optional[str] = None,
    ):
        self._name = name
        self._description = description
        self._parameters_schema = parameters_schema
        self.media_directory = media_directory
        self.default_audio_url = default_audio_url
        self.speaker_name_field = speaker_name_field
        self.default_speaker_name = default_speaker_name
        self.district_field = district_field
        self.default_district = default_district
        self.experience_years_field = experience_years_field
        self.default_experience_years = default_experience_years
        self.client_event_type = client_event_type
        self.success_status = success_status
        self.not_found_status = not_found_status
        self.success_message_builder = success_message_builder
        self.fallback_narrate_builder = fallback_narrate_builder
        self._prompt_instruction = prompt_instruction or (
            f"• AUDIO MEDIA PLAYBACK RULE (`{self._name}`):\n"
            f"  1. Call `{self._name}` when the user asks to hear an audio clip, recording, or presentation.\n"
            "  2. Give a brief, courteous 1-sentence introduction, then stop speaking immediately so the audio file can play."
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return self._parameters_schema

    @property
    def prompt_instruction(self) -> Optional[str]:
        return self._prompt_instruction


    async def execute(self, args: Dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        speaker = args.get(self.speaker_name_field) or self.default_speaker_name
        district = args.get(self.district_field) or self.default_district
        audio_url = args.get("audio_url") or self.default_audio_url
        years = args.get(self.experience_years_field) or self.default_experience_years

        filename = audio_url.replace("\\", "/").split("/")[-1]
        filepath = os.path.join(self.media_directory, filename)

        info = {
            "speaker": speaker,
            "district": district,
            "audio_url": audio_url,
            "experience_years": years,
        }

        if os.path.isfile(filepath):
            print(f"🎙️ [Audio Playback Tool] Found '{filename}' for '{speaker}' ({district})")
            client_evt = {
                "type": self.client_event_type,
                "farmer_name": speaker,
                "district": district,
                "audio_url": audio_url,
                "experience_years": years,
            }
            if self.success_message_builder:
                msg = self.success_message_builder(info)
            else:
                msg = f"{speaker} ({district}) નો ઓડિયો અનુભવ AI બોલવાનું પૂરું કરે પછી શરૂ થશે."

            return ToolResult(
                status=self.success_status,
                message=msg,
                client_event=client_evt,
                queued_audio_url=audio_url,
            )
        else:
            print(f"ℹ️ [Audio Playback Tool] Audio file not found at: {filepath} — falling back to AI narration.")
            if self.fallback_narrate_builder:
                fallback_msg = self.fallback_narrate_builder(info)
            else:
                fallback_msg = (
                    f"ધ્યાન આપો: {speaker} ({district}) માટે કોઈ રેકોર્ડ કરેલી ઓડિયો કે વિડીયો ફાઇલ ઉપલબ્ધ નથી! "
                    f"તેથી ઓડિયો પ્લે થશે નહીં. "
                    f"કૃપા કરીને તમે પોતે જ તમારા અવાજમાં અત્યારે જ આ ખેડૂત ({speaker}, જિલ્લો: {district}) નો સંપૂર્ણ જાત-અનુભવ "
                    f"— પાક, પ્રાકૃતિક પદ્ધતિઓ, ખર્ચમાં બચત, ચોખ્ખો નફો અને તેમની પ્રેરણાદાયી સલાહ — સ્નેહપૂર્વક અને વિગતવાર બોલીને ખેડૂતને સંભળાવો."
                )

            return ToolResult(
                status=self.not_found_status,
                message=fallback_msg,
                queued_audio_url=None,
            )
