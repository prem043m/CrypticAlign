class MBTIEngine:

    MBTI_COMPATIBILITY = {

        "INTJ": {
            "ENFP": 100,
            "ENTP": 95,
            "INFJ": 85,
            "INTJ": 75
        },

        "INTP": {
            "ENTJ": 100,
            "ENFJ": 90,
            "INTP": 75
        },

        "INFJ": {
            "ENTP": 100,
            "ENFP": 95,
            "INFJ": 80
        },

        "ENFP": {
            "INTJ": 100,
            "INFJ": 95,
            "ENFP": 75
        }
    }

    @classmethod
    def get_score(
        cls,
        mbti_a,
        mbti_b
    ):

        if mbti_a == mbti_b:
            return 75

        return (
            cls.MBTI_COMPATIBILITY
            .get(mbti_a, {})
            .get(mbti_b, 50)
        )