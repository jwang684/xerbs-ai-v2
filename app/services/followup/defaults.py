from app.schemas.followup import FollowUpQuestion


def default_tcm_questions(language: str = "zh") -> list[FollowUpQuestion]:
    if not language.lower().startswith("zh"):
        return [
            FollowUpQuestion(id=1, question="Do you feel unusually cold or feverish?", options=["Cold", "Feverish", "Alternating", "Neither", "Unsure"], category="cold_heat"),
            FollowUpQuestion(id=2, question="How is your sweating?", options=["Normal", "Excessive", "Reduced", "None", "Night sweats"], category="sweat"),
            FollowUpQuestion(id=3, question="Any headache, dizziness, heaviness, or body aches?", options=["Headache", "Dizziness", "Heaviness", "Body aches", "None"], category="head_body"),
            FollowUpQuestion(id=4, question="How are your bowel movements?", options=["Normal", "Constipation", "Diarrhea", "Loose stool", "Blood present"], category="stool"),
            FollowUpQuestion(id=5, question="How is your appetite?", options=["Normal", "Low", "High", "Aversion to food", "Other"], category="diet"),
            FollowUpQuestion(id=6, question="Any chest or abdominal discomfort?", options=["None", "Chest tightness", "Chest pain", "Palpitations", "Shortness of breath"], category="chest_abdomen"),
            FollowUpQuestion(id=7, question="Any hearing, ringing, vision, or eye symptoms?", question_type="text", category="ears_eyes"),
            FollowUpQuestion(id=8, question="How is your thirst and drink preference?", options=["Normal", "Very thirsty", "Not thirsty", "Thirsty but little desire to drink", "Prefer cold drinks"], category="thirst"),
            FollowUpQuestion(id=9, question="Have you had a similar problem or important chronic illness before?", question_type="text", category="history"),
            FollowUpQuestion(id=10, question="Any likely trigger such as stress, diet, exertion, travel, or exposure?", question_type="text", category="trigger"),
        ]
    return [
        FollowUpQuestion(id=1, question="您是否有怕冷或发热的感觉？", options=["怕冷", "发热", "怕冷发热交替", "无明显寒热", "不确定"], category="寒热"),
        FollowUpQuestion(id=2, question="平时出汗情况如何？", options=["正常出汗", "出汗过多", "出汗过少", "不出汗", "盗汗"], category="汗"),
        FollowUpQuestion(id=3, question="是否有头痛、头晕、头重或身体酸痛？", options=["头痛", "头晕", "头重", "身体酸痛", "无"], category="头身"),
        FollowUpQuestion(id=4, question="大便情况如何？", options=["正常", "便秘", "腹泻", "便溏", "便血"], category="二便"),
        FollowUpQuestion(id=5, question="食欲如何？", options=["正常", "食欲不振", "食欲亢进", "厌食", "其他"], category="饮食"),
        FollowUpQuestion(id=6, question="胸腹部是否有不适？", options=["无", "胸闷", "胸痛", "心悸", "气短"], category="胸腹"),
        FollowUpQuestion(id=7, question="耳目方面是否有耳鸣、听力或视力变化？", question_type="text", category="耳目"),
        FollowUpQuestion(id=8, question="口渴及饮水偏好如何？", options=["正常", "口渴", "不渴", "渴不欲饮", "渴喜冷饮"], category="渴饮"),
        FollowUpQuestion(id=9, question="以前是否有类似情况或重要慢性病史？", question_type="text", category="旧病"),
        FollowUpQuestion(id=10, question="发病前是否有情绪、饮食、劳累、旅行或受凉等诱因？", question_type="text", category="病因"),
    ]
