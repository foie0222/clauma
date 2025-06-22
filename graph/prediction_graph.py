"""
LangGraphを使った競馬予想対話システム
3人の専門家が議論して最終的な投資判断を下す
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from anthropic import Anthropic

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# .envファイルから環境変数を読み込み
load_dotenv()

from agents.race_expert import RaceExpert
from agents.jockey_expert import JockeyExpert
from agents.contrarian_expert import ContrarianExpert
from agents.moderator import Moderator


@dataclass
class PredictionState:
    """予想システムの状態"""
    race_info: str  # レース情報
    pace_expert_analysis: Optional[str] = None  # 展開予想専門家の分析
    jockey_expert_analysis: Optional[str] = None  # 騎手専門家の分析
    contrarian_expert_analysis: Optional[str] = None  # 穴狙い専門家の分析
    expert_opinions: List[str] = None  # 専門家意見リスト
    final_judgment: Optional[Dict] = None  # 最終判断
    is_complete: bool = False  # 完了フラグ
    
    def __post_init__(self):
        if self.expert_opinions is None:
            self.expert_opinions = []


class HorseRacePredictionGraph:
    """競馬予想対話グラフ"""
    
    def __init__(self, anthropic_client: Optional[Anthropic] = None):
        self.client = anthropic_client or Anthropic()
        self.pace_expert = RaceExpert(self.client)
        self.jockey_expert = JockeyExpert(self.client)
        self.contrarian_expert = ContrarianExpert(self.client)
        self.moderator = Moderator(self.client)
        
        # グラフの構築
        self.graph = self._build_graph()
    
    def _build_graph(self) -> CompiledStateGraph:
        """LangGraphを構築"""
        
        workflow = StateGraph(PredictionState)
        
        # ノードの追加
        workflow.add_node("pace_analysis", self._pace_expert_analysis)
        workflow.add_node("jockey_analysis", self._jockey_expert_analysis)
        workflow.add_node("contrarian_analysis", self._contrarian_expert_analysis)
        workflow.add_node("make_judgment", self._final_judgment)
        
        # エッジの設定（順次実行・討議なし）
        workflow.set_entry_point("pace_analysis")
        workflow.add_edge("pace_analysis", "jockey_analysis")
        workflow.add_edge("jockey_analysis", "contrarian_analysis")
        workflow.add_edge("contrarian_analysis", "make_judgment")
        workflow.add_edge("make_judgment", END)
        
        return workflow.compile()
    
    def _pace_expert_analysis(self, state: PredictionState) -> PredictionState:
        """展開予想専門家の初期分析"""
        
        opinion = self.pace_expert.analyze_race(state.race_info)
        analysis_text = f"【展開予想専門家】\n{opinion.analysis}\n推奨馬: {opinion.recommended_horses}\n確信度: {opinion.confidence:.2f}\n根拠: {opinion.reasoning}"
        
        state.pace_expert_analysis = analysis_text
        state.expert_opinions.append(analysis_text)
        
        return state
    
    def _jockey_expert_analysis(self, state: PredictionState) -> PredictionState:
        """騎手専門家の初期分析"""
        
        opinion = self.jockey_expert.analyze_race(state.race_info)
        analysis_text = f"【騎手専門家】\n{opinion.analysis}\n推奨馬: {opinion.recommended_horses}\n確信度: {opinion.confidence:.2f}\n根拠: {opinion.reasoning}"
        
        state.jockey_expert_analysis = analysis_text
        state.expert_opinions.append(analysis_text)
        
        return state
    
    def _contrarian_expert_analysis(self, state: PredictionState) -> PredictionState:
        """穴狙い専門家の分析"""
        
        opinion = self.contrarian_expert.analyze_race(state.race_info)
        analysis_text = f"【穴狙い専門家】\n{opinion.analysis}\n推奨馬: {opinion.recommended_horses}\n確信度: {opinion.confidence:.2f}\n根拠: {opinion.reasoning}"
        
        state.contrarian_expert_analysis = analysis_text
        state.expert_opinions.append(analysis_text)
        
        return state
    
    def _final_judgment(self, state: PredictionState) -> PredictionState:
        """最終判断"""
        
        final_judgment = self.moderator.make_final_judgment(
            state.race_info,
            state.pace_expert_analysis,
            state.jockey_expert_analysis,
            state.contrarian_expert_analysis
        )
        
        # 結果を辞書形式で保存
        state.final_judgment = {
            "consensus_analysis": final_judgment.consensus_analysis,
            "minority_opinions": final_judgment.minority_opinions,
            "expert_reliability": final_judgment.expert_reliability,
            "summary": final_judgment.summary,
            "recommendations": [
                {
                    "horse_number": rec.horse_number,
                    "win_odds": rec.win_odds,
                    "expected_value": rec.expected_value,
                    "bet_amount": rec.bet_amount,
                    "confidence": rec.confidence,
                    "edge_score": rec.edge_score
                }
                for rec in final_judgment.recommendations
            ],
            "reasoning": final_judgment.reasoning,
            "risk_assessment": final_judgment.risk_assessment
        }
        
        state.is_complete = True
        
        return state
    
    def predict_race(self, race_info: str) -> Dict[str, Any]:
        """レース予想を実行"""
        
        # 初期状態の設定
        initial_state = PredictionState(
            race_info=race_info,
            expert_opinions=[]
        )
        
        # グラフの実行
        result = self.graph.invoke(initial_state)
        
        # 結果の整理
        return {
            "race_info": race_info,
            "expert_opinions": result["expert_opinions"],
            "final_judgment": result["final_judgment"]
        }


def main():
    # 情報をファイルから読み込み
    data_path = os.path.join(os.path.dirname(__file__), "../data/race.txt")
    
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            sample_race = f.read()
    except FileNotFoundError:
        print(f"エラー: {data_path} が見つかりません")
        return
    
    # 予想システムの実行
    prediction_system = HorseRacePredictionGraph()
    result = prediction_system.predict_race(sample_race)
    
    print("=== 競馬予想システム実行結果 ===")
    print("\n=== 専門家意見 ===")
    for i, opinion in enumerate(result['expert_opinions'], 1):
        print(f"{i}. {opinion}\n")
    
    print("=== 最終判断 ===")
    if result['final_judgment']:
        judgment = result['final_judgment']
        print(f"コンセンサス分析: {judgment['consensus_analysis']}")
        print(f"マイノリティ意見: {judgment['minority_opinions']}")
        print(f"専門家信頼度: {judgment['expert_reliability']}")
        print(f"総合分析: {judgment['summary']}")
        print(f"推奨ベット:")
        for rec in judgment['recommendations']:
            print(f"  {rec['horse_number']}番 オッズ{rec['win_odds']} 期待値{rec['expected_value']:.2f} 金額{rec['bet_amount']}円 エッジスコア{rec['edge_score']:.2f}")
        print(f"判断根拠: {judgment['reasoning']}")
        print(f"リスク評価: {judgment['risk_assessment']}")


if __name__ == "__main__":
    main()
    