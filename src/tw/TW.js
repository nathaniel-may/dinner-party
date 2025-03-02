'use strict;'

import { twMerge as twMergeJS} from 'tailwind-merge'

export const twMerge = function (arr) {
    return twMergeJS(...arr)
}