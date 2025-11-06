import React from 'react';
import { IconCpu } from './Icons.jsx';

const GeminiInsights = ({ summary, suggestedQuestions, onQuestionClick, isSummarizing }) => (
    <div className="flex items-start gap-4 my-6">
        <div className="w-10 h-10 rounded-full bg-gray-700 flex items-center justify-center flex-shrink-0">
            <IconCpu className="text-blue-300" />
        </div>
        <div className="flex-grow">
            <p className="font-bold text-gray-300 mb-1">✨ Gemini Insights</p>
            <div className="bg-gray-800 p-4 rounded-xl text-gray-300 space-y-4">
                <p>{summary}</p>
                {!isSummarizing && suggestedQuestions.length > 0 && (
                    <div>
                        <p className="font-semibold mb-2">Suggested Questions:</p>
                        <div className="flex flex-col items-start gap-2">
                            {suggestedQuestions.map((q, i) => (
                                <button
                                    key={i}
                                    onClick={() => onQuestionClick(q)}
                                    className="text-left text-blue-400 hover:text-blue-300 transition-colors bg-gray-700/50 px-3 py-2 rounded-lg w-full"
                                >
                                    {q}
                                </button>
                            ))}
                        </div>
                    </div>
                )}
            </div>
        </div>
    </div>
);

export default GeminiInsights;

