'use strict;'

export const replaceChildren = function (elem) {
    return function (elems) {
        return function () {
            elem.replaceChildren(...elems);
        };
    };
}

export const outerHTML = function (elem) {
    return function () {
        return elem.outerHTML;
    };
}