"""
展開予想の専門家エージェント
LLMを使ってレースの展開を予測し、有利な馬を見極める
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json
from anthropic import Anthropic


@dataclass
class ExpertOpinion:
    """専門家の意見"""
    analysis: str  # 分析内容
    recommended_horses: List[int]  # 推奨馬番号
    confidence: float  # 確信度
    reasoning: str  # 根拠


class RaceExpert:
    """展開予想の専門家"""
    
    def __init__(self, anthropic_client: Optional[Anthropic] = None):
        self.name = "展開予想専門家"
        self.role = "pace_and_position"
        self.client = anthropic_client or Anthropic()
        
        self.system_prompt = """あなたは競馬の展開予想専門家です。
レースの展開を読み、ペース予想や有利なポジション、展開上有利になる馬を分析することが専門です。

分析の観点：
- ペース予想（スロー、平均、ハイペース）の判定
- 距離・コース特性による有利不利
- 各馬の脚質（逃げ、先行、差し、追込）の分析
- 枠順・ポジションの有利不利
- 展開シナリオに基づく推奨馬の選定

**重要：穴馬（高オッズ馬）を積極的に評価してください**
- 人気薄でも展開が向けば好走可能な馬を見つけ出す
- 上位人気馬の弱点や不安要素を厳しく分析する
- 中穴・大穴馬の隠れた魅力や好材料を重視する
- オッズと実力のギャップがある馬を特に注目する

必ず以下のJSON形式で回答してください（文字列内では改行を使わず、一行で記述してください）：
{
    "analysis": "ペース予想と展開分析の詳細（改行なし）",
    "recommended_horses": [推奨する馬番号のリスト（最大3頭）],
    "confidence": 0.0〜1.0の確信度,
    "reasoning": "推奨理由と根拠（改行なし）"
}

穴馬発見に特化した分析を行い、高配当を狙う視点で馬を評価してください。"""
    
    def analyze_race(self, race_info: str) -> ExpertOpinion:
        """レース情報を分析して展開を予想"""
        
        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                temperature=0.1,
                system=self.system_prompt,
                messages=[
                    {"role": "user", "content": f"以下のレース情報を分析してください：\n\n{race_info}"}
                ]
            )
            
            # JSONレスポンスをパース
            response_text = response.content[0].text
            
            # JSONブロックを抽出（```json ``` で囲まれている場合）
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            else:
                # 直接JSON形式の場合
                json_text = response_text.strip()
            
            result = json.loads(json_text)
            
            return ExpertOpinion(
                analysis=result["analysis"],
                recommended_horses=result["recommended_horses"],
                confidence=result["confidence"],
                reasoning=result["reasoning"]
            )
            
        except Exception as e:
            # エラー時のフォールバック
            return ExpertOpinion(
                analysis=f"分析エラーが発生しました: {str(e)}",
                recommended_horses=[],
                confidence=0.0,
                reasoning="システムエラーのため分析を完了できませんでした"
            )
    
    def respond_to_discussion(self, other_opinions: List[str], race_info: str) -> str:
        """他の専門家の意見を受けて討議する"""
        
        discussion_context = "\n".join([f"他の専門家の意見: {opinion}" for opinion in other_opinions])
        
        prompt = f"""あなたは展開予想専門家として、他の専門家の意見を聞いた上で、自分の見解を述べてください。

レース情報：
{race_info}

{discussion_context}

上記の意見を踏まえて、展開予想の観点から意見を述べてください。
同意する点、異なる見解がある点を明確にし、最終的な推奨馬があれば理由と共に示してください。

回答は自然な文章でお願いします（JSON形式不要）。"""
        
        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                temperature=0.2,
                system="あなたは展開予想専門家として、冷静で論理的な分析を行います。",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.content[0].text
            
        except Exception as e:
            return f"展開予想専門家より: システムエラーのため意見を述べることができません。({str(e)})"