package spinach.sentence;

import org.apache.commons.lang.builder.EqualsBuilder;
import org.apache.commons.lang.builder.HashCodeBuilder;

/**
 * @author Calvin Huang
 *
 * Token in a sentence; stores information about itself
 * such as its form, lemma, part of speech.
 * Also stores syntactic relationship information, such as
 * the sentence index of its syntactic head and its relation
 * to said head.
 * In order to perform operations such as looking up children
 * and adjacent tokens quickly in a sentence, the index of this
 * token is kept in itself as well.
 */
public class Token {

    public final String form;
    public final String lemma;
    public final String pos;
    public final String syntacticHeadRelation;

    public final int headSentenceIndex;
    public final int sentenceIndex;

    public Token(String form, String lemma,
                 String pos, String semanticHeadRelation,
                 int headSentenceIndex, int sentenceIndex){

        this.form = form;
        this.lemma = lemma;
        this.pos = pos;
        this.syntacticHeadRelation = semanticHeadRelation;
        this.headSentenceIndex = headSentenceIndex;
        this.sentenceIndex = sentenceIndex;
    }

    public boolean equals(Object o){
        if (o == null)
            return false;
        if (o == this)
            return true;
        if (o.getClass() != getClass())
            return false;

        Token t = (Token) o;
        return new EqualsBuilder().
                append(form, t.form).
                append(lemma, t.lemma).
                append(pos, t.pos).
                append(syntacticHeadRelation, t.syntacticHeadRelation).
                append(headSentenceIndex, t.headSentenceIndex).
                append(sentenceIndex, t.sentenceIndex).
                isEquals();
    }

    public int hashCode(){
        return new HashCodeBuilder(73, 23).
                append(form).
                append(lemma).
                append(pos).
                append(syntacticHeadRelation).
                append(headSentenceIndex).
                append(sentenceIndex).hashCode();
    }

    public static Token emptyToken = new Token("", "", "", "", -1, -1);
}
