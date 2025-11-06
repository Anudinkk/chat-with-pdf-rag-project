import React from 'react';
import { IconUser, IconCpu } from './Icons.jsx';

const ChatMessage = ({ sender, message, isInitial, onSummarize, isSummarizing }) => {
    const isUser = sender === 'user';
    return (
        <div className={`flex items-start gap-4 my-6`}>
            <div className={`w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0 ${isUser ? 'bg-indigo-800' : 'bg-gray-700'}`}>
                {isUser ? <IconUser className="text-indigo-300" /> : <IconCpu className="text-blue-300" />}
            </div>
            <div className="flex-grow">
                <p className="font-bold text-gray-300 mb-1">{isUser ? 'You' : 'AI Assistant'}</p>
                <div className="bg-gray-800 p-4 rounded-xl text-gray-300">
                    <p className="whitespace-pre-wrap">{message}</p>
                    {isInitial && (
                        <div className="mt-4 border-t border-gray-700 pt-3">
                            <button onClick={onSummarize} disabled={isSummarizing} className="flex items-center gap-2 text-blue-400 hover:text-blue-300 transition-colors disabled:opacity-50 disabled:cursor-wait">
                                <span className="text-xl">✨</span>
                                {isSummarizing ? 'Generating Insights...' : 'Generate Summary & Questions'}
                            </button>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default ChatMessage;

