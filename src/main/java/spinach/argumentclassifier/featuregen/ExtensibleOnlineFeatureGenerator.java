package spinach.argumentclassifier.featuregen;

import com.google.common.collect.Sets;
import edu.stanford.nlp.util.ErasureUtils;
import spinach.argumentclassifier.ArgumentClassifier;
import spinach.sentence.SemanticFrameSet;
import spinach.sentence.Token;

import java.io.*;
import java.util.*;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

/**
 * This feature generator, in addition to the argument feature generator,
 * enables addition of user-defined custom features
 *
 * @author Calvin Huang
 */
public class ExtensibleOnlineFeatureGenerator extends ArgumentFeatureGenerator {

    private Set<IndividualFeatureGenerator> enabledFeatures;
    private Set<IndividualFeatureGenerator> featureGeneratorList =
            new HashSet<IndividualFeatureGenerator>();

    @Override
    protected Collection<String> featuresOf(SemanticFrameSet sentenceAndPredicates,
                                            Token argument, Token predicate) {

        Collection<String> features = super.featuresOf(sentenceAndPredicates, argument, predicate);
        List<Token> featureTokens = new ArrayList<Token>();
        /* featureToken tokens:
        0: argument
        1: predicate
        2: ppHead
         */

        featureTokens.add(argument);
        featureTokens.add(predicate);

        Token ppHead;
        if (argument.syntacticHeadRelation.equals("PMOD"))
            ppHead = sentenceAndPredicates.getLeftSiblings(argument).getLast();
        else
            ppHead = sentenceAndPredicates.getParent(argument);
        featureTokens.add(ppHead);


        for (IndividualFeatureGenerator featureGenerator : enabledFeatures) {
            Collection<String> newFeatures = featureGenerator.
                    featuresOf(sentenceAndPredicates, featureTokens);
            if (newFeatures != null)
                features.addAll(newFeatures);
        }

        return features;

    }

    /**
     * Enables a certain feature
     *
     * @param featureGenerator feature to enable
     * @return whether or not feature was actually added (may have already been present)
     */
    public boolean enableFeatureType(IndividualFeatureGenerator featureGenerator) {
        return enabledFeatures.add(featureGenerator);
    }

    /**
     * Disables a certain feature
     *
     * @param featureGenerator feature to disable
     * @return whether or not feature was actually disable (may not have been present)
     */
    public boolean disableFeatureType(IndividualFeatureGenerator featureGenerator) {
        return enabledFeatures.remove(featureGenerator);
    }

    public int numFeatureTypes() {
        return featureGeneratorList.size();
    }

    /**
     * Gives the number of enabled additional features
     *
     * @return number of enabled extra features
     */
    public int numAddlFeatures() {
        return enabledFeatures.size();
    }

    /**
     * Disables all extra features
     */
    public void clearFeatures() {
        enabledFeatures.clear();
    }

    /**
     * Adds some featuregenerator to the list of feature generators
     *
     * @param featureGenerator feature generator to be added
     * @return index of feature generator in list
     */
    public int addFeature(IndividualFeatureGenerator featureGenerator) {
        featureGeneratorList.add(featureGenerator);
        return featureGeneratorList.size() - 1;
    }

    private void addDefaultFeatures() {
        addFeature(new IndividualFeatureGenerator() {
            //0: existSemDeprel
            @Override
            public Collection<String> featuresOf(SemanticFrameSet frameSet, List<Token> featureTokens) {
                //TODO
                return null;
            }
        });

        addFeature(new IndividualFeatureGenerator() {
            String identifier = "previousArgClass";

            @Override
            public Collection<String> featuresOf(SemanticFrameSet frameSet, List<Token> featureTokens) {
                int mostRecentArgumentIndex = -1;
                String mostRecentLabel = null;

                Token predicate = featureTokens.get(1);
                for (Map.Entry<Token, String> pair : frameSet.argumentsOf(predicate).entrySet()) {
                    int currIndex = pair.getKey().sentenceIndex;
                    if (currIndex > mostRecentArgumentIndex) {
                        mostRecentArgumentIndex = currIndex;
                        mostRecentLabel = pair.getValue();
                    }
                }

                if (mostRecentLabel == null)
                    return Collections.singletonList("previousArgClass:" + ArgumentClassifier.NIL_LABEL);
                else
                    return Collections.singletonList("previousArgClass:" + mostRecentLabel);
            }
        });
    }

    private void load(ObjectInputStream in) throws IOException, ClassNotFoundException {
        featureGeneratorList = ErasureUtils.uncheckedCast(in.readObject());
        enabledFeatures = ErasureUtils.uncheckedCast(in.readObject());

    }

    /**
     * Loads a feature generator from some file path.
     *
     * @param path path to saved feature generator
     * @return loaded feature generator
     * @throws ClassNotFoundException
     * @throws IOException
     */
    public static ExtensibleOnlineFeatureGenerator load(String path) throws ClassNotFoundException, IOException {
        GZIPInputStream is = new GZIPInputStream(new FileInputStream(path));

        ObjectInputStream in = new ObjectInputStream(is);
        ExtensibleOnlineFeatureGenerator ex = new ExtensibleOnlineFeatureGenerator();
        ex.load(in);
        in.close();
        is.close();
        return ex;
    }

    /**
     * Saves this feature generator to some file
     *
     * @param path file path to save to
     */
    public void save(String path) {
        try {
            ObjectOutputStream out = new ObjectOutputStream(new BufferedOutputStream(
                    new GZIPOutputStream(new FileOutputStream(path))));

            out.writeObject(featureGeneratorList);
            out.writeObject(enabledFeatures);

            out.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public Set<IndividualFeatureGenerator> featureGeneratorSet() {
        return Collections.unmodifiableSet(featureGeneratorList);
    }

    public void setFeatureGenerators(Set<IndividualFeatureGenerator> featureGenerators) {
        enabledFeatures = new HashSet<IndividualFeatureGenerator>(featureGenerators);
    }

    public Set<IndividualFeatureGenerator> enabledFeatures() {
        return Collections.unmodifiableSet(enabledFeatures);
    }

    public Set<IndividualFeatureGenerator> disabledFeatures() {
        return Sets.difference(featureGeneratorList, enabledFeatures).immutableCopy();
    }

}

