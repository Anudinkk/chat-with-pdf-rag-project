import React from 'react';
import { IconCpu, IconLoader } from './Icons.jsx';

const LoadingIndicator = () => (
    <div className="flex items-start gap-4 my-6">
        <div className="w-10 h-10 rounded-full bg-gray-700 flex items-center justify-center flex-shrink-0">
            <IconCpu className="text-blue-300" />
        </div>
        <div className="flex-grow">
             <p className="font-bold text-gray-300 mb-1">AI Assistant</p>
             <div className="bg-gray-800 p-4 rounded-xl flex items-center">
                 <IconLoader className="animate-spin text-blue-400 mr-3" />
                 <p className="text-gray-400">Thinking...</p>
             </div>
        </div>
    </div>
);

export default LoadingIndicator;

