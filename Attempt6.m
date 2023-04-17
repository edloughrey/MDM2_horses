clear all
format long

marketWin = readtable('dwbfpricesukwin29032023.csv');
marketPlace = readtable('dwbfpricesukplace29032023.csv');

sortedWin = sortrows(marketWin, 1);
sortedPlace = sortrows(marketPlace, 1);

eventID_w = sortedWin.EVENT_ID;     eventID_p = sortedPlace.EVENT_ID;
uniqueID_w = unique(eventID_w);     uniqueID_p = unique(eventID_p);
numID_w = length(uniqueID_w);       numID_p = length(uniqueID_p);


% clear up the win market table (remove any 2 horse races)
idx_w = [];
for i = 1:numID_w
    TableWin{i} = sortedWin(eventID_w == uniqueID_w(i), :);
    Tw = TableWin{i};
    if height(Tw) > 2
        idx_w = [idx_w; 1];
    else
        idx_w = [idx_w; 0];
    end
end


% Combine all tables with inx_w=1 into a big table
T = table();
for i = 1:length(idx_w)
    if idx_w(i) == 1
        T = cat(1, T, TableWin{i});
    end
end


% redefine win market variables now we've filtered the table
eventID_w = T.EVENT_ID;
uniqueID_w = unique(eventID_w);
numID_w = length(uniqueID_w);


% this loop separates all races into their own tables
% info for race i will be found in Table(i)
for i = 1:numID_w    
    TableWin{i} = T(eventID_w == uniqueID_w(i), :);
    TableWin{i} = sortrows(TableWin{i}, 5);
    Tw = TableWin{i};
    BSP = Tw.BSP;
    Probs{i} = 1 ./ Tw.BSP;
    if length(Probs{i}) > 6
        k{i} = 3;
    else
        k{i} = 2;
    end
end
probs_k = [Probs; k];


% Here is the same as above but for the place market tables, but we don't
% need to worry about the k values
for i = 1:numID_p    
    TablePlace{i} = sortedPlace(eventID_p == uniqueID_p(i), :);
    TablePlace{i} = sortrows(TablePlace{i}, 5);
    Tp = TablePlace{i};
    BSP = Tp.BSP;
    ProbsP{i} = 1 ./ Tp.BSP;
end
% compareProbs = [Probs; ProbsP];
% probs are good


if numID_w ~= numID_p
    % Find indices of elements in cell2 that have no matching dimensions in cell1
    indices_to_remove = [];
    for i = 1:numel(TableWin)
        size_i = size(TableWin{i});
        found_match = false;
        for j = 1:numel(TablePlace)
            if isequal(size_i, size(TablePlace{j}))
                found_match = true;
                break;
            end
        end
        if ~found_match
            indices_to_remove = [indices_to_remove i];
        end
    end
    % Remove elements from cell2 that have no matching dimensions in cell1
    TableWin(indices_to_remove) = [];
    Probs(indices_to_remove) = [];
end
numID_w = numID_p;


% calc_probs = zeros(numID_w, 20);
for u = 1:numID_w
    probbys = Probs{u};
    nHorses = length(probbys);
    info = probs_k(:, u);
    kVal = cell2mat(info(2));

    calc_probs{u} = zeros(1, nHorses);

    for x = 1:nHorses
        % swap horse x with final horse
        probbys([x nHorses]) = probbys([nHorses x]);

        % Function handles for q
        q1 = @(i) 1 - probbys(i);
        q2 = @(i, j) 1 - probbys(i) - probbys(j);
        q3 = @(i, j, k) 1 - probbys(i) - probbys(j) - probbys(k);
        
        % Function handles for P
        P3 = @(i, j, k) (probbys(i)*probbys(j)*probbys(k)) / (q1(i)*q2(i, j));
        P2 = @(i, j) (probbys(i)*probbys(j)) / (q1(i));

        P2s = [];
        P3s = [];
        perms = nchoosek(1:(nHorses-1), 2);
        nPerms = size(perms, 1);
                
        if kVal == 2
            for i = 1:(nHorses-1)
                P2s = [P2s; P2(nHorses, i)];
                P2s = [P2s; P2(i, nHorses)];
            end
            odds_place = sum(P2s);
        else
            for i = 1:(nPerms)
                P3s = [P3s; P3(nHorses, perms(i,1), perms(i,2))];
                P3s = [P3s; P3(nHorses, perms(i,2), perms(i,1))];
                P3s = [P3s; P3(perms(i,1), nHorses, perms(i,2))];
                P3s = [P3s; P3(perms(i,2), nHorses, perms(i,1))];
                P3s = [P3s; P3(perms(i,1), perms(i,2), nHorses)];
                P3s = [P3s; P3(perms(i,2), perms(i,1), nHorses)];
            end
            odds_place = sum(P3s);
        end
        % calc_probs(u, x) = odds_place;    
        calc_probs{u}(x) = odds_place;
        % t(x) = odds_place;
    end
end


% % Make a numID x 20 matrix for betfair's place market probabilites
% betfair_probs = zeros(numID_w, 20);
% for i = 1:numID_p
%     matProbsP = cell2mat(ProbsP(i));
%     raceSize = length(matProbsP);
%     for u = 1:raceSize
%         betfair_probs(i, u) = matProbsP(u);
%     end
% end


% Make a 1 x nHorses matrix for each race of betfairs probabilities
for i = 1:numID_p
    nHorses = length(Probs{i});
    betfair_probs{i} = zeros(1, nHorses);
    matProbsP = cell2mat(ProbsP(i));
    raceSize = length(matProbsP);
    for u = 1:raceSize
        betfair_probs{i}(u) = matProbsP(u);
    end
end


corr_coeffs = [];
figure
for z = 1:numID_p
    scatter(betfair_probs{z}, calc_probs{z}, 'red', 'filled')
    hold on
    corr_coeff = corrcoef(betfair_probs{z}, calc_probs{z});
    corr_coeffs = [corr_coeffs corr_coeff(2)];
end


score = mean(corr_coeffs);
sprintf('Correlation coefficient = %.2f', score)
% scatter(betfair_probs, calc_probs, 'red', 'filled')
xlabel("Betfair's Place Market 1/BSP")
ylabel("Harville's predicted odds")
title("Harville's model against Betfair's Model")

