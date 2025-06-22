"""
穴狙い専門の逆張り派エージェント
人気薄の馬から隠れた魅力を見つけ出すことに特化
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


class ContrarianExpert:
    """穴狙い専門の逆張り派"""
    
    def __init__(self, anthropic_client: Optional[Anthropic] = None):
        self.name = "穴狙い専門家"
        self.role = "contrarian_analysis"
        self.client = anthropic_client or Anthropic()
        
        self.system_prompt = """あなたは競馬の穴狙い専門家です。
人気薄の馬から隠れた魅力を見つけ出し、高配当を狙うことに特化した分析を行います。

分析の観点：
- 人気馬の過大評価ポイントを見抜く
- 不人気馬の過小評価要因を発見する
- 前走の不利や展開の悪さなど、能力以外の敗因を重視
- 距離・コース替わりでの激変可能性
- 調教での変化や仕上がりの向上
- 騎手交代によるプラス要因
- 馬場状態による恩恵を受ける可能性
- 血統的な底力や成長の余地
- 人気薄ゆえの好枠順の活用
- 少頭数や展開向きなどの好条件

**重要な指針：**
- オッズ10倍以上の馬を中心に分析
- 人気馬の弱点を積極的に探す
- 一般的な予想の盲点を突く
- データに表れない好材料を重視
- 「みんなが見落としている点」を探す

必ず以下のJSON形式で回答してください（文字列内では改行を使わず、一行で記述してください）：
{
    "analysis": "穴馬発見の分析内容（改行なし）",
    "recommended_horses": [推奨する穴馬の馬番号リスト（最大3頭、オッズ10倍以上推奨）],
    "confidence": 0.0〜1.0の確信度,
    "reasoning": "穴馬推奨の具体的根拠（改行なし）"
}

逆張りの視点で、市場が見落としている投資機会を発見してください。"""
    
    def analyze_race(self, race_info: str) -> ExpertOpinion:
        """レース情報を分析して穴馬を発見"""
        
        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                temperature=0.3,  # 少し高めの温度で創造的な分析を促す
                system=self.system_prompt,
                messages=[
                    {"role": "user", "content": f"以下のレース情報から穴馬を発見してください：\n\n{race_info}"}
                ]
            )
            
            # JSONレスポンスをパース
            response_text = response.content[0].text
            
            # JSONブロックを抽出
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            else:
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
                analysis=f"穴馬分析エラーが発生しました: {str(e)}",
                recommended_horses=[],
                confidence=0.0,
                reasoning="システムエラーのため分析を完了できませんでした"
            )