# 記事レビュー

## 記事タイトル
{{ article_title }}

## 記事内容
```markdown
{{ article_content }}
```

## タスク
上記の技術記事をレビューし、改善点を指摘してください。

以下の観点でチェックしてください：
1. **技術的正確性**: 内容は正確か？誤解を招く記述はないか？
2. **可読性**: 専門用語は適切に説明されているか？初心者にもわかりやすいか？
3. **構成**: 論理的な流れになっているか？各セクションの役割は明確か？
4. **コードの品質**: コードは動作するか？ベストプラクティスに従っているか？
5. **実用性**: 読者が実際に手を動かして試せる内容か？

以下のJSON形式で回答してください：

```json
{
  "overall_score": 8,
  "needs_improvement": true,
  "summary": "全体的な評価コメント",
  "strengths": [
    "良い点1",
    "良い点2"
  ],
  "improvements": [
    {
      "section_index": 0,
      "issue": "問題点の説明",
      "suggestion": "改善の提案"
    }
  ],
  "code_issues": [
    {
      "section_index": 2,
      "issue": "コードの問題",
      "suggestion": "修正案"
    }
  ]
}
```

## 注意点
- overall_score は 1-10 のスコア（8以上なら needs_improvement は false）
- section_index は 0 から始まるセクション番号
- 改善点がない場合は improvements を空配列にしてください
- 致命的な問題がなければ needs_improvement は false にしてください
