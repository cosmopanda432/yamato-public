"""
言語処理層 — 岩戸隠れアーキテクチャ

天岩戸神話に基づく自然言語理解・応答生成パイプライン。
yamatoLLM の3層アーキテクチャにおける言語処理層を担う。

5章構造:
    第一章: 参集 (天安河原)      — 入力埋め込み
    第二章: 思案 (思兼神)        — 意図解析・ルーティング
    第三章: 奉献 (布刀玉命+真榊) — 知識統合 (RAG)
    第四章: 神楽 (天宇受売命)    — 生成・感情制御
    第五章: 開戸 (天手力男神)    — 出力確定

接続:
    言依さし (Kotoyosashi) — コード生成層との接続プロトコル
    忌部 (Inbe)           — 入出力浄化
"""

from .yasukawara_embedding import YasukawaraEmbedding
from .omoikane_intent import OmoikaneIntentRouter
from .futodama_retriever import FutodamaRetriever
from .amenouzume_decoder import AmenouzumeDecoder
from .tajikarao_output import TajikraoOutput
from .kotoyosashi_protocol import KotoyosashiProtocol
from .inbe_sanitizer import InbeSanitizer

__all__ = [
    "YasukawaraEmbedding",
    "OmoikaneIntentRouter",
    "FutodamaRetriever",
    "AmenouzumeDecoder",
    "TajikraoOutput",
    "KotoyosashiProtocol",
    "InbeSanitizer",
]
