package spinach.classify;

import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import spinach.argumentclassifier.ArgumentClassifier;
import spinach.argumentclassifier.featuregen.ExtensibleOnlineFeatureGenerator;
import spinach.argumentclassifier.featuregen.IndividualFeatureGenerator;
import spinach.predicateclassifier.PredicateClassifier;
import spinach.sentence.SemanticFrameSet;
import spinach.sentence.TokenSentence;
import spinach.sentence.TokenSentenceAndPredicates;

import java.util.*;

/**
 * A class that does the entire task for a sentence--
 * determines the predicates and the arguments of said sentence
 *
 * @author Calvin Huang
 */
public class StructuredClassifier implements GEN {

    private final ArgumentClassifier argumentClassifier;
    private final PredicateClassifier predicateClassifier;
    private final int epochs;

    private ExtensibleOnlineFeatureGenerator featureGenerator;

    private Collection<SemanticFrameSet> trainingFrames;
    private Collection<SemanticFrameSet> testingFrames;

    /**
     * Generates a structured classifier with a certain argument classifier, predicate classifier,
     * and a certain number of epochs
     *
     * @param argumentClassifier  ArgumentClassifier to use
     * @param predicateClassifier PredicateClassifier to use
     * @param epochs              number of times to iterate through training datasets
     */
    public StructuredClassifier(ArgumentClassifier argumentClassifier,
                                PredicateClassifier predicateClassifier, int epochs) {
        this.argumentClassifier = argumentClassifier;
        this.predicateClassifier = predicateClassifier;
        this.epochs = epochs;
    }

    /**
     * Generates a structured classifier with a certain argument classifier, predicate classifier,
     * and runs 10 epochs during training
     *
     * @param argumentClassifier  ArgumentClassifier to use
     * @param predicateClassifier PredicateClassifier to use
     */
    public StructuredClassifier(ArgumentClassifier argumentClassifier,
                                PredicateClassifier predicateClassifier) {
        this(argumentClassifier, predicateClassifier, 10);
    }

    /**
     * Given a sentence, determines the predicates and arguments for that sentence
     *
     * @param sentence sentence to analyze
     * @return SemanticFrameSet with the original sentence, along with predicates and arguments
     */
    public SemanticFrameSet parse(TokenSentence sentence) {

        TokenSentenceAndPredicates sentenceAndPredicates =
                predicateClassifier.sentenceWithPredicates(sentence);

        return argumentClassifier.framesWithArguments(sentenceAndPredicates);

    }

    private SemanticFrameSet trainingParse(TokenSentence sentence) {

        TokenSentenceAndPredicates sentenceAndPredicates =
                predicateClassifier.trainingSentenceWithPredicates(sentence);

        return argumentClassifier.trainingFramesWithArguments(sentenceAndPredicates);

    }

    /**
     * Trains this structured classifier on a collection of known SemanticFrameSets
     *
     * @param goldFrames SemanticFrameSets with known good semantic data
     */
    public void train(Collection<SemanticFrameSet> goldFrames) {
        setTrainingFrames(goldFrames);
        train();
    }


    private void train() {
        for (int i = 0; i < epochs; i++) {

            List<SemanticFrameSet> goldFramesCopy = new ArrayList<SemanticFrameSet>(trainingFrames);

            Collections.shuffle(goldFramesCopy, new Random(i));

            for (SemanticFrameSet goldFrame : goldFramesCopy)
                train(goldFrame);
        }
    }

    private void train(SemanticFrameSet goldFrame) {

        //when this is run, parse ignores the predicates and semantic data
        SemanticFrameSet predictedFrame = trainingParse(goldFrame);

        predictedFrame.trimPredicates();

        predicateClassifier.update(predictedFrame, goldFrame);
        argumentClassifier.update(predictedFrame, goldFrame);
    }

    public void setTrainingFrames(Collection<SemanticFrameSet> goldFrames) {
        trainingFrames = goldFrames;
    }

    /**
     * Trains the argument feature generator for this object's ArgumentClassifier--
     * enables features that increase the F1 of the structured classifier
     *
     * @param trainingFrames SemanticFrameSets with known good semantic data, to train on
     * @param testingFrames  SemanticFrameSets with known good semantic data, to test against
     */
    public void trainArgumentFeatureGenerator(List<SemanticFrameSet> trainingFrames,
                                              List<SemanticFrameSet> testingFrames) {
        this.trainingFrames = trainingFrames;
        this.testingFrames = testingFrames;

        if (argumentClassifier.isFeatureTrainable())
            featureGenerator = (ExtensibleOnlineFeatureGenerator) argumentClassifier.getFeatureGenerator();
        else {
            System.err.println("Cannot train feature generator--is not a trainable feature generator");
            return;
        }

        featureGenerator.clearFeatures();

        Random random = new Random();
        for (IndividualFeatureGenerator i : featureGenerator.featureGeneratorSet())    //random subset of features
            if (random.nextBoolean())
                featureGenerator.enableFeatureType(i);

        Set<IndividualFeatureGenerator> featureGeneratorSet = featureGenerator.enabledFeatures();

        while (true) {
            Set<IndividualFeatureGenerator> additions = recruitMore(featureGeneratorSet);
            if (additions.isEmpty())
                return;

            Set<IndividualFeatureGenerator> updatedFeatureGeneratorSet = shakeOff(
                    Sets.union(featureGeneratorSet, additions));

            if (argumentTrainAndScore(featureGeneratorSet) > argumentTrainAndScore(updatedFeatureGeneratorSet))
                return;

            featureGeneratorSet = updatedFeatureGeneratorSet;
        }
    }

    private double argumentTrainAndScore(Set<IndividualFeatureGenerator> featureGenerators) {
        argumentClassifier.reset();
        predicateClassifier.reset();
        featureGenerator.setFeatureGenerators(featureGenerators);
        train();
        return new Metric(this, testingFrames).argumentF1s().getCount(
                Metric.TOTAL);
    }

    private Set<IndividualFeatureGenerator> recruitMore(Set<IndividualFeatureGenerator> featureGenerators) {
        Set<IndividualFeatureGenerator> additions = new HashSet<IndividualFeatureGenerator>();
        Set<IndividualFeatureGenerator> possibleAdditions = featureGenerator.disabledFeatures();

        double originalScore = argumentTrainAndScore(featureGenerators);

        for (IndividualFeatureGenerator possibleAddition : possibleAdditions)
            if (argumentTrainAndScore(Sets.union(featureGenerators, Collections.singleton(possibleAddition))) > originalScore)
                additions.add(possibleAddition);

        return additions;
    }

    private Set<IndividualFeatureGenerator> shakeOff(Set<IndividualFeatureGenerator> featureGeneratorSet) {
        Set<IndividualFeatureGenerator> maxFeatureGenerators =
                new HashSet<IndividualFeatureGenerator>(featureGeneratorSet);

        while (true) {
            Set<IndividualFeatureGenerator> originalFeatureGenerators =
                    new HashSet<IndividualFeatureGenerator>(maxFeatureGenerators);
            Map<IndividualFeatureGenerator, Double> unsortedFeatureGenerators =
                    new HashMap<IndividualFeatureGenerator, Double>();

            Set<IndividualFeatureGenerator> currentFeatureGenerators =
                    new HashSet<IndividualFeatureGenerator>(originalFeatureGenerators);

            for (IndividualFeatureGenerator generator : originalFeatureGenerators) {
                currentFeatureGenerators.remove(generator);
                unsortedFeatureGenerators.put(generator, argumentTrainAndScore(currentFeatureGenerators));
                currentFeatureGenerators.add(generator);
            }


            double sMaxScore = 0;
            boolean maxScoreUpToDate = false;
            for (Set<IndividualFeatureGenerator> sortedFeatureGenerators =
                         invertedSortByValues(unsortedFeatureGenerators).keySet();
                 !sortedFeatureGenerators.isEmpty();
                 sortedFeatureGenerators.remove(Iterables.getFirst(sortedFeatureGenerators, null))) {

                if (!maxScoreUpToDate) {
                    sMaxScore = argumentTrainAndScore(maxFeatureGenerators);
                    maxScoreUpToDate = true;
                }

                if (argumentTrainAndScore(sortedFeatureGenerators) > sMaxScore) {
                    maxFeatureGenerators =
                            new HashSet<IndividualFeatureGenerator>(sortedFeatureGenerators);
                    maxScoreUpToDate = false;
                }
            }
            if (originalFeatureGenerators.equals(maxFeatureGenerators))
                return originalFeatureGenerators;
        }
    }

    private static <K, V extends Comparable<V>> Map<K, V> invertedSortByValues(final Map<K, V> map) {
        Comparator<K> valueComparator = new Comparator<K>() {
            public int compare(K k1, K k2) {
                int compare = map.get(k2).compareTo(map.get(k1));
                if (compare == 0) return 1;
                else return -compare;
            }
        };
        Map<K, V> sortedByValues = new TreeMap<K, V>(valueComparator);
        sortedByValues.putAll(map);
        return new LinkedHashMap<K, V>(sortedByValues);
    }

}
