package spinach.classify;

import com.google.common.collect.Sets;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.Counters;
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

    private static final boolean PREDICTED_PRED_DURING_ARG_TESTING = true;

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

        correctArguments.clear();
        predictedArguments.clear();
        goldArguments.clear();

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


            if (PREDICTED_PRED_DURING_ARG_TESTING) {
                for (Token predictedPredicate : predictedPredicates)
                    for (Map.Entry<Token, String> entry : predictedFrameSet.argumentsOf(predictedPredicate).entrySet())
                        predictedArguments.incrementCount(entry.getValue());

                for (Token goldPredicate : goldPredicates)
                    for (Map.Entry<Token, String> entry : goldFrameSet.argumentsOf(goldPredicate).entrySet())
                        goldArguments.incrementCount(entry.getValue());

                for (Token goldAndPredictedPredicate : goldAndPredictedPredicates)
                    for (Map.Entry<Token, String> correctEntry :
                            Sets.intersection(goldFrameSet.argumentsOf(goldAndPredictedPredicate).entrySet(),
                                    predictedFrameSet.argumentsOf(goldAndPredictedPredicate).entrySet()))
                        correctArguments.incrementCount(correctEntry.getValue());

            } else {

                SemanticFrameSet argPredictedFrameSet = gen.argParse(goldFrameSet);

                for (Token goldPredicate : goldPredicates) {
                    for (Map.Entry<Token, String> entry : goldFrameSet.argumentsOf(goldPredicate).entrySet())
                        goldArguments.incrementCount(entry.getValue());

                    for (Map.Entry<Token, String> entry : argPredictedFrameSet.argumentsOf(goldPredicate).entrySet())
                        predictedArguments.incrementCount(entry.getValue());

                    for (Map.Entry<Token, String> correctEntry :
                            Sets.intersection(goldFrameSet.argumentsOf(goldPredicate).entrySet(),
                                    predictedFrameSet.argumentsOf(goldPredicate).entrySet()))
                        correctArguments.incrementCount(correctEntry.getValue());
                }
            }

            goldArguments.setCount(TOTAL, Counters.sumEntries(goldArguments, goldArguments.keySet()));
            correctArguments.setCount(TOTAL, Counters.sumEntries(correctArguments, correctArguments.keySet()));
            predictedArguments.setCount(TOTAL, Counters.sumEntries(predictedArguments, predictedArguments.keySet()));
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
        return harmMean(predicatePrecision(), predicateRecall());
    }

    /**
     * Num. of correctly predicted predicates.
     *
     * @return number of correctly predicted predicates
     */
    public int predicateCorrect() {
        return correctPredicateNum;
    }

    /**
     * Num. of predicted predicates.
     *
     * @return number of predicted predicates
     */
    public int predicatePredicted() {
        return predictedPredicateNum;
    }

    /**
     * Num. of gold predicates.
     *
     * @return number of predicates in the gold frame sets
     */
    public int predicateGold() {
        return goldPredicateNum;
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
        Counter<String> precisions = argumentPrecisions();
        Counter<String> recalls = argumentRecalls();

        for (String s : correctArguments.keySet())
            f1s.incrementCount(s, harmMean(precisions.getCount(s), recalls.getCount(s)));
        return f1s;
    }

    private double harmMean(double d1, double d2) {
        if (d1 == 0.0 || d2 == 0.0)
            return 0;
        return 2 / ((1.0 / d1) + (1.0 / d2));
    }

}
