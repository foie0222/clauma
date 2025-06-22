"""
総合判断専門家（モデレーター）
展開予想専門家と騎手専門家の意見を統合し、最終的な投資判断を行う
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import json
from anthropic import Anthropic


@dataclass
class BettingRecommendation:
    """ベッティング推奨"""
    horse_number: int  # 馬番号
    win_odds: float  # 単勝オッズ
    expected_value: float  # 期待値
    bet_amount: int  # ベット額（1000円固定）
    confidence: float  # 確信度
    edge_score: float = 0.0  # エッジスコア（0.0-1.0）


@dataclass
class FinalJudgment:
    """最終判断"""
    consensus_analysis: str  # 専門家コンセンサスの要約
    minority_opinions: str  # マイノリティ意見とエッジ評価
    expert_reliability: Dict[str, float]  # 各専門家の信頼度スコア
    summary: str  # 総合分析サマリー
    recommendations: List[BettingRecommendation]  # ベッティング推奨
    reasoning: str  # 判断根拠
    risk_assessment: str  # リスク評価


class Moderator:
    """総合判断専門家（モデレーター）"""
    
    def __init__(self, anthropic_client: Optional[Anthropic] = None):
        self.name = "総合判断専門家"
        self.role = "final_judge"
        self.client = anthropic_client or Anthropic()
        
        self.system_prompt = """あなたは競馬投資の総合判断専門家です。
展開予想専門家、騎手専門家、穴狙い専門家の3人の意見を統合し、期待値に基づく投資判断を行います。

**重要な役割：専門家コンセンサスとマイノリティ意見の分析**
1. 3人の専門家の意見から「コンセンサス（多数派意見）」を抽出
2. 「マイノリティ意見（逸脱意見）」を特定し、そのエッジを評価
3. 各専門家の意見の信頼度スコアを推定（0.0-1.0）
4. オッズとのギャップから投資機会を発見

判断基準：
- 期待値1.0超えの馬をすべて推奨対象とする
- 1bet = 1000円で統一
- 単勝のみで勝負
- コンセンサス意見とマイノリティ意見のバランスを考慮
- 「エッジの効いた意見」を特に重視

**分析の観点：**
- 3人全員が推す馬：高い信頼度だがオッズは低い可能性
- 2人が推す馬：バランス型の投資対象
- 1人だけが強く推す馬：エッジが効いており高配当の可能性
- 誰も推さないがオッズが高い馬：見落とし馬の可能性も検討

期待値計算：
期待値 = (勝率 × オッズ × 信頼度加重) / 1.0

必ず以下のJSON形式で回答してください（文字列内では改行を使わず、一行で記述してください）：
{
    "consensus_analysis": "専門家コンセンサスの要約（改行なし）",
    "minority_opinions": "マイノリティ意見とそのエッジ評価（改行なし）",
    "expert_reliability": {
        "pace_expert": 信頼度スコア,
        "jockey_expert": 信頼度スコア,
        "contrarian_expert": 信頼度スコア
    },
    "summary": "総合的な分析結果（改行なし）",
    "recommendations": [
        {
            "horse_number": 馬番号,
            "win_odds": 単勝オッズ,
            "expected_value": 期待値,
            "bet_amount": 1000,
            "confidence": 確信度,
            "edge_score": エッジスコア（0.0-1.0）
        }
    ],
    "reasoning": "投資判断の根拠（改行なし）",
    "risk_assessment": "リスク評価（改行なし）"
}

市場が見落としている投資機会を発見し、期待値の高い馬を推奨してください。"""
    
    def make_final_judgment(self, race_info: str, pace_expert_opinion: str, 
                          jockey_expert_opinion: str, contrarian_expert_opinion: str) -> FinalJudgment:
        """最終判断を下す"""
        
        prompt = f"""以下の情報を基に、最終的な投資判断を行ってください。

レース情報：
{race_info}

展開予想専門家の意見：
{pace_expert_opinion}

騎手専門家の意見：
{jockey_expert_opinion}

穴狙い専門家の意見：
{contrarian_expert_opinion}

3人の専門家の意見を分析し、コンセンサスとマイノリティ意見を特定してください。
特に「エッジの効いた意見」に注目し、市場が見落としている投資機会を発見してください。
期待値1.0を超える馬があれば、すべて推奨してください。"""
        
        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=3000,
                temperature=0.1,
                system=self.system_prompt,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # JSONレスポンスをパース
            response_text = response.content[0].text
            
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            else:
                json_text = response_text.strip()
            
            result = json.loads(json_text)
            
            # BettingRecommendationオブジェクトに変換
            recommendations = []
            for rec in result["recommendations"]:
                recommendations.append(BettingRecommendation(
                    horse_number=rec["horse_number"],
                    win_odds=rec["win_odds"],
                    expected_value=rec["expected_value"],
                    bet_amount=rec["bet_amount"],
                    confidence=rec["confidence"],
                    edge_score=rec.get("edge_score", 0.0)
                ))
            
            return FinalJudgment(
                consensus_analysis=result["consensus_analysis"],
                minority_opinions=result["minority_opinions"],
                expert_reliability=result["expert_reliability"],
                summary=result["summary"],
                recommendations=recommendations,
                reasoning=result["reasoning"],
                risk_assessment=result["risk_assessment"]
            )
            
        except Exception as e:
            # エラー時のフォールバック
            return FinalJudgment(
                consensus_analysis="エラーのため分析不可",
                minority_opinions="エラーのため分析不可",
                expert_reliability={"pace_expert": 0.0, "jockey_expert": 0.0, "contrarian_expert": 0.0},
                summary=f"最終判断エラーが発生しました: {str(e)}",
                recommendations=[],
                reasoning="システムエラーのため判断を完了できませんでした",
                risk_assessment="エラーのためリスク評価不可"
            )
