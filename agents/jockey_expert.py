"""
騎手専門家エージェント
LLMを使って騎手の実績・相性・調子などを総合分析する
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


class JockeyExpert:
    """騎手専門家"""
    
    def __init__(self, anthropic_client: Optional[Anthropic] = None):
        self.name = "騎手専門家"
        self.role = "jockey_analysis"
        self.client = anthropic_client or Anthropic()
        
        self.system_prompt = """あなたは競馬の騎手専門家です。
騎手のあらゆる要素を分析し、騎手の視点から有力馬を見極めることが専門です。

分析の観点（考えうるすべての要素）：
- 騎手の通算成績・実績・タイトル
- 最近の調子・勝率・連対率
- コース別・距離別・馬場別の得意/不得意
- 芝・ダート・重馬場での成績
- 当該馬との騎乗歴・相性
- 初騎乗か継続騎乗か
- 騎手の脚質・戦術の得意分野
- 減量騎手（☆▲△）の恩恵
- 騎手の年齢・経験・安定性
- 騎手の所属厩舎との関係
- 騎手の体重・騎乗スタイル
- 大レース・重賞での実績
- 同期・同世代騎手との比較
- 騎手の怪我・休養明けかどうか
- その他すべての騎手に関わる要素

**重要：穴馬（高オッズ馬）の可能性を積極的に探してください**
- 若手騎手や減量騎手の勢いや可能性を重視する
- 人気薄馬でも騎手の特性が活かせるケースを見つける
- 騎手と馬の相性で化ける可能性のある組み合わせを評価する
- ベテラン騎手の穴馬での一発勝負を見逃さない
- オッズに反映されていない騎手の隠れた実力を発見する

必ず以下のJSON形式で回答してください（文字列内では改行を使わず、一行で記述してください）：
{
    "analysis": "騎手要素の総合分析（改行なし）",
    "recommended_horses": [推奨する馬番号のリスト（最大3頭）],
    "confidence": 0.0〜1.0の確信度,
    "reasoning": "推奨理由と根拠（改行なし）"
}

穴馬発見に特化した騎手分析を行い、高配当につながる組み合わせを見つけてください。"""
    
    def analyze_race(self, race_info: str) -> ExpertOpinion:
        """レース情報を分析して騎手の観点から予想"""
        
        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                temperature=0.1,
                system=self.system_prompt,
                messages=[
                    {"role": "user", "content": f"以下のレース情報を騎手の観点から徹底分析してください：\n\n{race_info}"}
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
                analysis=f"騎手分析エラーが発生しました: {str(e)}",
                recommended_horses=[],
                confidence=0.0,
                reasoning="システムエラーのため分析を完了できませんでした"
            )
    
    def respond_to_discussion(self, other_opinions: List[str], race_info: str) -> str:
        """他の専門家の意見を受けて討議する"""
        
        discussion_context = "\n".join([f"他の専門家の意見: {opinion}" for opinion in other_opinions])
        
        prompt = f"""あなたは騎手専門家として、他の専門家の意見を聞いた上で、自分の見解を述べてください。

レース情報：
{race_info}

{discussion_context}

上記の意見を踏まえて、騎手分析の観点から意見を述べてください。
同意する点、異なる見解がある点を明確にし、最終的な推奨馬があれば理由と共に示してください。

回答は自然な文章でお願いします（JSON形式不要）。"""
        
        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                temperature=0.2,
                system="あなたは騎手専門家として、騎手のあらゆる要素を考慮した分析を行います。",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.content[0].text
            
        except Exception as e:
            return f"騎手専門家より: システムエラーのため意見を述べることができません。({str(e)})"