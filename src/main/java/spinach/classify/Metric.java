package spinach.classify;

import com.google.common.collect.Sets;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import spinach.sentence.SemanticFrameSet;
import spinach.sentence.Token;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;

/**
 * A class that generates statistics for classifiers, i.e.
 * comparing output with known values and calculating their
 * precision, recall, F1 scores, etc.
 *
 * @author Calvin Huang
 */
public class Metric {

    public final static String TOTAL = "TOTAL";

    private int correctPredicateNum;
    private int goldPredicateNum;
    private int predictedPredicateNum;

    private Counter<String> correctArguments = new ClassicCounter<String>();
    private Counter<String> predictedArguments = new ClassicCounter<String>();
    private Counter<String> goldArguments = new ClassicCounter<String>();

    private final Collection<SemanticFrameSet> goldFrameSets;
    private final GEN gen;

    /**
     * Generates a metric from a classifier and a bunch of known good SemanticFrameSets
     *
     * @param g             predicate/argument classifier for token sentence
     * @param goldFrameSets gold SemanticFrameSets to compare GEN output against
     */
    public Metric(GEN g, Collection<SemanticFrameSet> goldFrameSets) {
        gen = g;
        this.goldFrameSets = goldFrameSets;
        recalculateScores();
    }

    /**
     * Recalculates the scores if the GEN has been modified
     */
    public void recalculateScores() {
        correctPredicateNum = 0;
        goldPredicateNum = 0;
        predictedPredicateNum = 0;

        correctArguments = new ClassicCounter<String>();
        predictedArguments = new ClassicCounter<String>();
        goldArguments = new ClassicCounter<String>();

        for (SemanticFrameSet goldFrameSet : goldFrameSets) {
            SemanticFrameSet predictedFrameSet = gen.parse(goldFrameSet);

            List<Token> goldPredicates = goldFrameSet.getPredicateList();
            List<Token> predictedPredicates = predictedFrameSet.getPredicateList();
            List<Token> goldAndPredictedPredicates = new ArrayList<Token>();

            for (Token goldPredicate : goldPredicates)
                if (predictedPredicates.contains(goldPredicate))
                    goldAndPredictedPredicates.add(goldPredicate);

            predictedPredicateNum += predictedPredicates.size();
            goldPredicateNum += goldPredicates.size();
            correctPredicateNum += goldAndPredictedPredicates.size();

            for (Token predictedPredicate : predictedPredicates) {
                for (Map.Entry<Token, String> entry : predictedFrameSet.argumentsOf(predictedPredicate).entrySet()) {
                    predictedArguments.incrementCount(entry.getValue());
                    predictedArguments.incrementCount(TOTAL);
                }
            }

            for (Token goldPredicate : goldPredicates) {
                for (Map.Entry<Token, String> entry : goldFrameSet.argumentsOf(goldPredicate).entrySet()) {
                    goldArguments.incrementCount(entry.getValue());
                    goldArguments.incrementCount(TOTAL);
                }
            }

            for (Token goldAndPredictedPredicate : goldAndPredictedPredicates) {
                for (Map.Entry<Token, String> correctEntry :
                        Sets.intersection(goldFrameSet.argumentsOf(goldAndPredictedPredicate).entrySet(),
                                predictedFrameSet.argumentsOf(goldAndPredictedPredicate).entrySet())) {
                    correctArguments.incrementCount(correctEntry.getValue());
                    correctArguments.incrementCount(TOTAL);
                }
            }
        }
    }

    /**
     * Gives the precision of the predicate classifier,
     * i.e. correct predicates / predicted predicates
     *
     * @return predicate classifier precision
     */
    public double predicatePrecision() {
        double precision = ((double) correctPredicateNum) / ((double) predictedPredicateNum);
        if (Double.isNaN(precision))
            return 0;
        return precision;
    }

    /**
     * Gives the recall of the predicate classifier,
     * i.e. correct predicates / gold predicates
     *
     * @return predicate classifier recall
     */
    public double predicateRecall() {
        double recall = ((double) correctPredicateNum) / ((double) goldPredicateNum);
        if (Double.isNaN(recall))
            return 0;
        return recall;
    }

    /**
     * Gives the F1 score of the predicate classifier,
     * i.e. harmonic mean of its precision and recall
     *
     * @return predicate classifier F1 score
     */
    public double predicateF1() {
        double f1 = 2 / ((1 / predicatePrecision()) + (1 / predicateRecall()));
        if (Double.isNaN(f1))
            return 0;
        return f1;
    }

    /**
     * Gives the precision of the argument classifier split by label,
     * i.e. correct arguments / predicted arguments
     *
     * @return Counter of labels, with the precision of each label
     */
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

    /**
     * Gives the recall of the argument classifier split by label,
     * i.e. correct arguments / gold arguments
     *
     * @return Counter of labels, with the recall of each label
     */
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

    /**
     * Gives the F1 score of the argument classifier split by label,
     * i.e. harmonic mean of recall and precision
     *
     * @return Counter of labels, with the F1 of each label
     */
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
