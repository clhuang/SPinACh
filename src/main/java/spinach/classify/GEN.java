package spinach.classify;

import spinach.sentence.SemanticFrameSet;
import spinach.sentence.TokenSentence;

/**
 * @author Calvin Huang
 *
 * Interface for a class that takes in a token sentence,
 * and generates a semantic frameset from it
 */
public interface GEN {
    public SemanticFrameSet parse(TokenSentence sentence);
}
