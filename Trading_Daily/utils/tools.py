def MajorityPrediction(*predictions):
    win_count = 0
    lose_count = 0
    for pred in predictions:
        if pred == 'Win':
            win_count += 1
        elif pred == 'Lose':
            lose_count += 1
    if win_count > lose_count:
        return 'Win'
    return 'Lose'
    
def UnanimityPrediction(*predictions):
    for pred in predictions:
        if pred != 'Win':
            return 'Lose'
    return 'Win'

def SafeDivide(numerator, denominator):
    return (numerator / denominator) * 100 if denominator != 0 else 0