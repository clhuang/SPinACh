package spinach.classify;

import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import spinach.sentence.SemanticFrameSet;
import spinach.sentence.Token;

import java.util.Collection;
import java.util.List;
import java.util.Map;

public class Metric {

    public final static String TOTAL = "TOTAL";

    private int correctPredicateNum = 0;
    private int goldPredicateNum = 0;
    private int predictedPredicateNum = 0;

    private Counter<String> correctArguments = new ClassicCounter<String>();
    private Counter<String> predictedArguments = new ClassicCounter<String>();
    private Counter<String> goldArguments = new ClassicCounter<String>();

    public Metric(GEN g, Collection<SemanticFrameSet> goldFrameSets) {
        for (SemanticFrameSet goldFrameSet : goldFrameSets) {
            SemanticFrameSet predictedFrameSet = g.parse(goldFrameSet);

            List<Token> goldPredicates = goldFrameSet.getPredicateList();
            List<Token> predictedPredicates = predictedFrameSet.getPredicateList();
            for (Token predictedPredicate : predictedPredicates) {
                predictedPredicateNum++;
                if (goldPredicates.contains(predictedPredicate)) {
                    correctPredicateNum++;
                    for (Map.Entry<Token, String> entry : goldFrameSet.argumentsOf(predictedPredicate).entrySet()) {
                        correctArguments.incrementCount(entry.getValue());
                        correctArguments.incrementCount(TOTAL);
                    }
                }
                for (Map.Entry<Token, String> entry : predictedFrameSet.argumentsOf(predictedPredicate).entrySet()) {
                    predictedArguments.incrementCount(entry.getValue());
                    predictedArguments.incrementCount(TOTAL);
                }
            }

            for (Token goldPredicate : goldPredicates) {
                goldPredicateNum++;
                for (Map.Entry<Token, String> entry : goldFrameSet.argumentsOf(goldPredicate).entrySet()) {
                    goldArguments.incrementCount(entry.getValue());
                    goldArguments.incrementCount(TOTAL);
                }
            }
        }
    }

    public double predicatePrecision() {
        double precision = ((double) correctPredicateNum) / ((double) predictedPredicateNum);
        if (Double.isNaN(precision))
            return 0;
        return precision;
    }

    public double predicateRecall() {
        double recall = ((double) correctPredicateNum) / ((double) goldPredicateNum);
        if (Double.isNaN(recall))
            return 0;
        return recall;
    }

    public double predicateF1() {
        double f1 = 2 / ((1 / predicatePrecision()) + (1 / predicateRecall()));
        if (Double.isNaN(f1))
            return 0;
        return f1;
    }

    public Counter<String> argumentPrecisions() {
        Counter<String> precisions = new ClassicCounter<String>();
        for (String s : correctArguments.keySet()) {
            double precision = correctArguments.getCount(s) / predictedArguments.getCount(s);
            if (Double.isNaN(precision))
                precision = 0;
            precisions.incrementCount(s, precision);
        }
        return precisions;
    }

    public Counter<String> argumentRecalls() {
        Counter<String> recalls = new ClassicCounter<String>();
        for (String s : correctArguments.keySet()) {
            double recall = correctArguments.getCount(s) / goldArguments.getCount(s);
            if (Double.isNaN(recall))
                recall = 0;
            recalls.incrementCount(s, recall);
        }
        return recalls;
    }

    public Counter<String> argumentF1s() {
        Counter<String> f1s = new ClassicCounter<String>();
        for (String s : correctArguments.keySet()) {
            double f1 = 2 / ((1 / argumentPrecisions().getCount(s)) +
                    (1 / argumentRecalls().getCount(s)));
            if (Double.isNaN(f1))
                f1 = 0;
            f1s.incrementCount(s, f1);
        }
        return f1s;
    }

}
