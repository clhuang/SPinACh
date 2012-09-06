package spinach.classify;

import spinach.sentence.SemanticFrameSet;
import spinach.sentence.TokenSentence;

/**
 * Interface for a class that takes in a token sentence,
 * and generates a semantic frameset from it
 *
 * @author Calvin Huang
 */
public interface GEN {
    /**
     * Given a sentence of tokens, determines the predicates of that sentence and the arguments of said sentence
     *
     * @param sentence sentence to analyze
     * @return SemanticFrameSet with the original sentence, along with predicates and arguments
     */
    public SemanticFrameSet parse(TokenSentence sentence);
}
