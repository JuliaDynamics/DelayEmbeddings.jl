function [Y_tot,tau_vals,epsilon_mins,gammas] = pecora_embed(varargin)
% PECORA_EMBED is a unified approach to properly embed a time series based
% on the paper of Pecora et al., Chaos 17 (2007).
%
% Minimum input-arguments: 1
% Maximum input-arguments: 7
%
% [Y, taus, epsilon_mins, gammas] = pecora_embed(x,tau_max,...
%                                   epsilon_tries,datasample,...
%                                   theiler_window,norm,break_percentage);
%
% This function embeds the input time series 'x' with different delay
% times tau. The approach views the problem of choosing all embedding
% parameters as being one and the same problem addressable using a single 
% statistical test formulated directly from the reconstruction theorems. 
% This allows for varying time delays appropriate to the data and 
% simultaneously helps decide on embedding dimension. A second new 
% statistic, undersampling, acts as a check against overly long time delays 
% and overly large embedding dimension.
%
% Input:    
%
% 'x'               A uni- or multivariate time series, which needs to be
%                   embedded. If the input data is a multivariate set, the
%                   algorithm scans all time series and chooses the time
%                   delays and time series for the reconstruction
%                   automatically.
% 'tau_max'         Defines up to which maximum delay time tau the
%                   algorithm shall look (Default is tau = 50).
% 'epsilion_tries'  Defines how many epsilon refinements are made for the
%                   computation of the continuity stastistic. Specifically
%                   the range of the input time series is divided by this 
%                   input number in order to achieve the epsilon 
%                   refinements the algorithms tests (Default is 20).
% 'datasample'      Defines the size of the random phase space vector 
%                   sample the algorithm considers for each tau value, in 
%                   order to compute the continuity statistic. This is a
%                   float from the intervall (0 1]. The size of the 
%                   considered sample is 'datasample'*length of the current
%                   phase space trajectory (Default is 0.5, i.e. half of 
%                   the trajectory points will be considered).
% 'theiler_window'  Defines a temporal correlation window for which no
%                   nearest neighbours are considered to be true, since
%                   they could be direct predecessors or successors of the
%                   fiducial point (Default is 1).
% 'norm'            norm for distance calculation in phasespace. Set to
%                   'euc' (euclidic) or 'max' (maximum). Default is Maximum
%                   norm.
% 'break_percentage'is the fraction of the standard deviation of the
%                   continuity statistic (of the first cycle of embedding),
%                   for which the algorithm breaks the computations.
%
%
%
% Output: (minimum 1, maximum 4)
%
% 'Y'               The embedded phase space trajectory            
% 'taus'            The (different) time delays chosen by the algorithm
% 'epsilon_mins'    Continuity statistic. A cell array storing all epsilon-
%                   mins as a function of 'taus' for each encountered 
%                   dimension, i.e. the size of 'epsilon_mins' is the same 
%                   as the final embedding dimension.
% 'gammas'          Undersampling statistic. A cell array storing all gamma
%                   values as a function of 'taus' for each encountered 
%                   dimension, i.e. the size of 'gammas' is the same 
%                   as the embedding final dimension.


% Copyright (c) 2019
% K. Hauke Kraemer, 
% Potsdam Institute for Climate Impact Research, Germany
% http://www.pik-potsdam.de
%
% This program is free software; you can redistribute it and/or
% modify it under the terms of the GNU General Public License
% as published by the Free Software Foundation; either version 2
% of the License, or any later version.

%% Assign input

% the input time series. Unlike the description in the docstring, yet there
% is just a univariate time series allowed
x = varargin{1};
% normalize time series
x = (x-mean(x))/std(x);

% make the input time series a column vector
if size(x,1)>size(x,2)
    x = x';
end
% as mentioned above, this will be changed in the end for allowing
% multivariate input
if size(x,1)~=1
    error('only univariate time series allowed for input')
end

try
    tau_max = varargin{2};
catch
    tau_max = 50;
end


try
    eps_tries = varargin{3};
catch
    eps_tries = 20;
end

try
    sample_size = varargin{4};
    if sample_size < 0 || sample_size > 1
        warning('break percentage input must be a value in the interval [0 1]')
        sample_size = 0.5;
    end
catch
    sample_size = 0.5;
end

try
    theiler = varargin{5};
catch
    theiler = 1;
end

    
methLib={'euc','max'}; % the possible norms
try
    norm = varargin{6};
    if ~isa(norm,'char') || ~ismember(norm,methLib)
       warning(['Specified norm should be one of the following possible values:',...
           10,sprintf('''%s'' ',methLib{:})])
       norm = 'max';
    end
catch
    norm = 'max';
end

try
    break_percentage = varargin{7};
    if break_percentage < 0 || break_percentage > 1
        warning('break percentage input must be a value in the interval [0 1]')
        break_percentage = 0.1;
    end
    
catch
    break_percentage = 0.1;
end


% set at which fraction the epsilon is supposed to start (e.g. 
% rangefactor = 2 means the first epsilon value is chosen to be 1/2 of the
% range of the input time series.)  I incorporated this in order to save
% computation time and "focus" on smaller scales for the epsilon statistic.
% This can be easily omitted in final implementation.
rangefactor = 1;

% confidence level for undersampling statistic (could also be input 
% parameter in final implementation)
alpha = 0.05;

% Matlab in- and output check
narginchk(1,7)
nargoutchk(1,4)

%% Start computation

% intial phase space vector (no embedding)
Y_old = x;

% set a flag, in order to tell the while loop when to stop. Each loop
% stands for encountering a new embedding dimension
flag = true;

% set index-counter for the while loop
cnt = 1;

% initial tau value for no embedding. This is trivial 0, when there is no
% embedding
tau_vals = 0;

% loop over increasing embedding dimensions until some break criterion will
% tell the loop to stop/break
while flag
     
    % preallocate storing vector for minimum epsilon vals (continuity 
    % statistic)
    epsilon_min_avrg = zeros(1,tau_max+1);
    
    % preallocate storing vector for gamma vals (undersampling statistic)
    gamma_avrg = zeros(1,tau_max+1);
    
    % set index-counter for the upcoming for-loop over the different tau 
    % values
    tau_counter = 1;
    % loop over the different tau values. Starting at 0 is important,
    % especially when considering a mutlivariate input data set to choose
    % from (in the final implemenation)
    for tau = 0:tau_max
        
        % create new phase space vector. 'embed2' is a helper function, 
        % which you find at the end of the script. At this point one could
        % incorporate onther loop over all the input time series, when
        % allowing multivariate input. 
        Y_new = embed2(Y_old,x,tau);    
        
        % select a random phase space vector sample. One could of course 
        % take all points from the actual trajectory here, but this would
        % be computationally overwhelming. This is why I incorporated the
        % input parameter 'sample_size' as the fraction of all phase space
        % points.
        data_samps = datasample(1:size(Y_new,1),floor(sample_size*size(Y_new,1)),...
            'Replace',false);
        
        % preallocate storing vector for minimum epsilon from the
        % continuity statistic
        epsilon_star = zeros(1,floor(sample_size*size(Y_new,1)));
        
        % preallocate storing vector for maximum gamma from the
        % undersampling statisctic
        gamma_k = zeros(1,floor(sample_size*size(Y_new,1)));
        
        % loop over all fiducial points, look at their neighbourhoods and
        % compute the continuity as well as the undersampling statistic for
        % each of them. Eventually we average over all the values to get
        % the continuity and undersampling statistic for the actual tau
        % value.
        for k = 1:floor(sample_size*size(Y_new,1))
            
            % bind the fiducial point from the trajectory sample
            fiducial_point = data_samps(k);
            
            % compute distances to all other points in dimension d. You'll
            % find the helper function at the end of the script
            distances = all_distances(Y_new(fiducial_point,1:end-1),...
                                                    Y_new(:,1:end-1),norm);
            
            % compute distances to all other points in dimension d+1 and 
            % also the componentwise distances for the undersampling 
            % statistic
            [distances2,comp_dist] = all_distances(Y_new(fiducial_point,:),...
                                                            Y_new,norm);
            
            % sort these distances in ascending order
            [~,ind] = sort(distances); 
            [~,ind2] = sort(distances2); 
            
            % 1) perform undersampling statistic test. You'll find the 
            % helper function undersampling at the end of this script.
            
            % herefore get the componentwise distances to compare against
            dist = comp_dist(ind2(2),:);
            % now run the undersampling statistic function on the first
            % component of the phase space vector and the "new" last
            % component, since the first component is the unshifted time
            % series. In case of a multivariate input signal this needs to
            % be done for all components, since they could possibly
            % originate from different input time series.
            [bo,gamma] = undersampling(Y_new(:,1),Y_new(:,end),dist,alpha);
                        
            % take the maximum gamma and the corresponding logical
            [gamma_k(k),ind3] = max(gamma);
            bool = bo(ind3);
            
            % 2) compute the continuity statistic
            
            % generate the possible epsilons
            % here the not really necessary 'rangefactor' I mentioned above
            % comes into play, in order to "focus" on the interesting,
            % smaller scales. As I said, this can just be omitted and does
            % not really change anything, unless 'eps_tries' is large
            % enough in order to provide a decent resolution
            epsilons = linspace(range(x)/rangefactor,0,eps_tries+1);

            % loop over all delta-neighbourhoods. Here we take the table
            % from the paper, which is based on an confidence level alpha =
            % 0.05. In the final implementation one should be able to
            % choose an alpha, maybe at least from 0.05 or 0.01. Therefore
            % one could look up a similar table (binomial distribution). We
            % try all these number of points, i.e. we try many different
            % delta's, as mentioned in the paper. For each of the delta
            % neighbourhoods we loop over the epsilons (decreasing values)
            % and stop, when we can not reject the null anymore. After
            % trying all deltas we pick the maximum epsilon from all
            % delta-trials. This is then the final epsilon for one specific
            % tau and one specific fiducial point. Afterwards we average
            % over all fiducial points.
            
            % table from the paper corresponding to alpha = 0.05
            delta_points = [5 6 7 8 9 10 11 12 13];
            epsilon_points = [5 6 7 7 8 9 9 9 10];
            
            % preallocate storing vector for the epsilons from which 
            % we cannot reject the null anymore (for each delta)
            epsilon_star_delta = zeros(1,length(delta_points));
            
            % loop over all the deltas (from the table above)
            for delta = 1:length(delta_points)
                
                neighbours = delta_points(delta); 
                neighbour_min = epsilon_points(delta);
                
                % loop over an decresing epsilon neighbourhood
                for epsi = 1:length(epsilons)-1
                    
                    % bind the actual neighbourhood-size
                    epsil = epsilons(epsi);

                    % define upper and lower epsilon neighbourhood bound,
                    % that is, the last component of the "new" embedding
                    % vector +- the epsilon neighbourhood. See Figure 1 in
                    % the paper
                    upper = Y_new(fiducial_point,size(Y_new,2))+epsil;
                    lower = Y_new(fiducial_point,size(Y_new,2))-epsil;

                    % scan neighbourhood of fiducial point and count the 
                    % projections of these neighbours, which fall into the 
                    % epsilon set

                    count = 0;
                    l = 2; % start with the first neighbour which is not 
                           % the fiducial point itself
                           
                    % loop over all neighbours (determined by the delta-
                    % neighbourhood-size) and count how many of those
                    % points fall within the epsilon neighbourhood.
                    for nei = 1:neighbours
                        
                        % this while loop gurantees, that we look at a true
                        % neighbour and not a one which lies in the
                        % correlation window of the fiducial point
                        while true
                            
                            % check whether the last component of this 
                            % neighbour falls into the epsilon set. 
                            % Therefore, first check that the neighbour is 
                            % not in the temporal correlation window 
                            % (determined by the input 'theiler').
                            % If it is, go on to the next.
                            if ind(l) > fiducial_point + theiler || ...
                                    ind(l) < fiducial_point - theiler
                                
                                % check if the valid neighbour falls in the
                                % epsilon neighbourhood. If it does, count
                                % it
                                if Y_new(ind(l),size(Y_new,2))<=upper &&...
                                        Y_new(ind(l),size(Y_new,2)) >=lower
                                    count = count + 1;
                                end
                                % go to the next neighbour
                                l = l + 1;
                                break
                            else
                                % check the next neighbour
                                l = l + 1;
                            end
                            % make sure the data set is sufficiently
                            % sampled, if not pass an error. Since 'ind' is
                            % a vector storing all indices of nearest
                            % neighbours and l is exceeding its length
                            % would mean that all neighbours are so close
                            % in time that they fall within the correlation
                            % window OR the number of neighbours one is
                            % looking at (see table 1 in the paper) is too
                            % large
                            if l > length(ind)
                                error('not enough neighbours')
                            end
                        end
                         
                    end               
                    
                    % if the number of neighbours from the delta
                    % neighbourhood, which get projected into the epsilon
                    % neighbourhood (Fig.1 in the paper) are smaller than
                    % the amount needed to reject the null, we break and
                    % store this particular epsilon value (which 
                    % determines the epsilon neighbourhood-size)
                    if count < neighbour_min
                        epsilon_star_delta(delta) = epsilons(epsi-1);
                        break
                    end
                end
                
            end
            
            % In the paper it is not clearly stated how to preceed here. We
            % are looking for the smallest scale for which we can not
            % reject the null. Since we can not prefer one delta
            % neighbourhood-size, we should take the maximum of all
            % smallest scales.
            
%             epsilon_star(k) = min(epsilon_star_delta);
            epsilon_star(k) = max(epsilon_star_delta);
        end
        
        % average over all fiducial points
        
        % continuity statistic
        epsilon_min_avrg(tau_counter) = mean(epsilon_star);
        % undersampling statistic
        gamma_avrg(tau_counter) = mean(gamma_k);
        
        % increase index counter for the tau value
        tau_counter = tau_counter + 1;
    end
    
    %%%% for the final implementation, where we allow for a multivariate
    %%%% input dataset we here need to perform the above procedure for ALL time
    %%%% series and then pick the one, where we find 'epsilon_min_avrg' is
    %%%% maximal.
    
    
    % save all epsilon min vals corresponding to the different tau-vals for
    % this dimension-iteration
    epsilon_mins{cnt} = epsilon_min_avrg;
    % save all gamma vals corresponding to the different tau-vals for
    % this dimension-iteration
    gammas{cnt} = gamma_avrg;
        
    
    % Now we have to determine the optimal tau value from the continuitiy
    % statistic. In the paper it is not clearly stated how to achieve
    % that. They state: "If possible we choose  at a local maximum of 
    % 'epsilon_min_avrg' to assure the most independent coordinates
    % as in Eq. 1. If 'epsilon_min_avrg' remains small out to large , we 
    % do not need to add more components;" 
    
    % So we decided to look first look for the local maxima (pks are the
    % total values and locs the corresponding indices):
    [pks,locs] = findpeaks(epsilon_min_avrg,'MinPeakDistance',2);
    % now we pick the first local maximum, for which the preceeding and
    % succeeding peak are are smaller.
    chosen_peak = 0;
    for i = 2:length(pks)-1
        if pks(i)>pks(i-1) && pks(i)>pks(i+1)
            % we save the chosen peak with its amplitude (pks) and its
            % index (locs)
            chosen_peak = [pks(i),locs(i)];
            break
        end
    end
    % If there has not been any peak chosen in the last for loop, we simply
    % take the maximum of all values.
    if ~chosen_peak
        % look for the largest one
        [~,maxind] = max(pks);
        % save the chosen peak with its amplitude (pks) and its
        % index (locs)
        chosen_peak = [pks(maxind),locs(maxind)];
    end
    
    % now assign the tau value to this peak, specifically to the
    % corresponding index
    tau_use = chosen_peak(2)-1; % minus 1, because 0 is included
    
    % construct phase space vector according to the tau value which
    % determines the local maximum of the statistic
    Y_old = embed2(Y_old,x,tau_use);
    
    % add the chosen tau value to the output variable
    tau_vals = [tau_vals tau_use];
    
    % break criterions (as mentioned, there is no criterion stated in the
    % paper. They say: "If 'epsilon_min_avrg' remains small out to large 
    % tau, we do not need to add more components; we are done and delta=d."
    % I have interpreted this in the following way: "remaining small out to
    % large tau means it has a vanishing variability, thus a small standard
    % variation (see Fig. 2 in the paper). Therefore I'll :
    
    % 1) break, if the standard deviation of the epsilon^*-curve right to
    % the chosen peak is less than break_percentage*reference_std, which is
    % the standard deviation right to the chosen peak for the very first 
    % epsilon^*-curve. 'break_percentage' is an input parameter at the
    % moment, but if one could find a "decent" value for this I would just
    % fix this and don't leave it to the user.
     
%     if cnt == 1 % i.e. dimension 1
%         
%         % compute std from epsilon-star curve right of the chosen maximum for
%         % the first curve
%         reference_curve = epsilon_min_avrg(chosen_peak(2):end);
%         reference_std = std(reference_curve);
%         
%     else
%         
%         % compute std from epsilon-star curve right of the chosen maximum 
%         curve = epsilon_min_avrg(chosen_peak(2):end);
%         curve_std = std(curve);
%         
%         % compare this standard deviation to the reference one
%         if curve_std < break_percentage*reference_std
%             flag = false;
%         end
%     end
    
    % 2) break, if the undersampling statistic cuts through the chosen alpha
    % level
    
%     if max(gamma_avrg) > alpha
%         flag = false;
%     end
    
    
    %%%% this is for testing the code, specifically for reproducing the 
    %%%% results shown in Fig. 2 and Fig. 5. We force to break after the 
    %%%% 4th embedding dimension has been reached
    if cnt == 4 
        flag = false;
    end
    
    % increase dimension index counter   
    cnt = cnt + 1;
end
% Output
Y_tot = Y_old;

end


%% Undersampling statistics function

function [bool,gamma] = undersampling(varargin)

% [bool,gamma] = undersampling(x1,x2,epsilon,alpha) computes the 
% undersampling statistic gamma for two input time series 'x1' and 'x2' of 
% the distance 'epsilon' under the confidence level 'alpha' (optional 
% input, Default: alpha=0.05).
% It is possible to input a vector 'epsilon', containing a number of
% distances. In this case, the output variables are of the same length as
% the 'epsilon'-vector. The logical output 'bool' determines whether the
% null hypothesis gets rejected under the confidence level 'alpha' 
% (bool = true) or not (bool = false).
%
% K.H.Kraemer, Mar 2020

%% Assign input 
x1 = varargin{1};
% normalize time series
x1 = (x1-mean(x1))/std(x1);

if size(x1,1)>size(x1,2)
    x1 = x1';
end
if size(x1,1)~=1
    error('only univariate time series allowed for input')
end

x2 = varargin{2};
% normalize time series
x2 = (x2-mean(x2))/std(x2);

if size(x2,1)>size(x2,2)
    x2 = x2';
end
if size(x2,1)~=1
    error('only univariate time series allowed for input')
end

epsilon = varargin{3};
if size(epsilon,1) ~= 1 && size(epsilon,2) ~=1
    error('provide a valid distance vector. - This is either a column or line vector.')
end
if length(epsilon) == 1
    dist_vec = false;
    if epsilon<0
        error('provide a valid distance to test against (positive float oder int)')
    end
else
    dist_vec = true;
    for i = 1:length(epsilon)
        if epsilon(i)<0
            error('provide a valid distance vector to test against (positive float oder int)')
        end
    end
end
try
    alpha = varargin{4};
    if alpha<0 || alpha>1
        error('choose a valid confidence level as a float from [0 1].')
    end
catch
    alpha = 0.05;
end


%% estimate probability density function from input time series and conv

if range(x1)>range(x2)
    % first time series
    [hist1, edges1] = histcounts(x1,'Normalization','pdf');

    % second time series
    [hist2, ~] = histcounts(x2,edges1,'Normalization','pdf');
else
    % first time series
    [hist2, edges2] = histcounts(x2,'Normalization','pdf');

    % second time series
    [hist1, edges1] = histcounts(x1,edges2,'Normalization','pdf');
end

% construct domains
binwidth1 = mean(diff(edges1));
xx1 = (edges1(1)+binwidth1/2):binwidth1:(edges1(end)-binwidth1/2);

% convolute these distributions
sigma = conv(hist1,hist2);
% normalize the distribution
sigma = sigma/sum(sigma);

% construct domain of the convoluted signal
xx2 = ((edges1(1)+binwidth1/2)-(floor(length(xx1)/2)*binwidth1)):binwidth1:...
    ((edges1(end)-binwidth1/2)+(floor(length(xx1)/2)*binwidth1));

% truncate domain by the last point due to the convolution
if size(xx2,2) == size(sigma,2) + 1
    xx2 = xx2(1:end-1);
end

% make a high resolution x-axis in the limits of the convolution support,
% in order to approximate the probabilities of finding a certain distance
xx22 = linspace(xx2(1),xx2(end),2*max(length(x1),length(x2)));

% interp the convolution signal to these high resolution points
sigma2 = interp1(xx2,sigma,xx22);
% normalize the distribution
sigma2 = sigma2/sum(sigma2);


%% probability to find a distance of less than or equal to epsilon

if dist_vec
    gamma = zeros(1,length(epsilon));
    bool = zeros(1,length(epsilon));
    for i = 1:length(epsilon)
        % find indices in the convolution signal corresponding to the 
        % distance input 'epsilon'
        n1 = find(xx22<epsilon(i));
        upper = n1(end);
        n2 = find(xx22>-epsilon(i));
        lower = n2(1);

        % compute gamma statistic
        gamma(i) = 0.5 * sum(sigma2(lower:upper));

        % compare to input confidence level
        if gamma(i) < alpha
            bool(i) = true;
        else 
            bool(i) = false;
        end
    end
    
else
    % find indices in the convolution signal corresponding to the distance
    % input 'epsilon'
    n1 = find(xx22<epsilon);
    upper = n1(end);
    n2 = find(xx22>-epsilon);
    lower = n2(1);

    % compute gamma statistic
    gamma = 0.5 * sum(sigma2(lower:upper));

    % compare to input confidence level
    if gamma < alpha
        bool = true;
    else 
        bool = false;
    end
 
end

end


%% Helper functions

function Y2 = embed2(varargin)
% embed2 takes a matrix 'Y' containing all phase space vectors, a univariate
% time series 'x' and a tau value 'tau' as input. embed2 then expands the 
% input phase space vectors by an additional component consisting of the 
% tau-shifted values of the input time series x.
% 
% Y2 = embed2(Y,x,tau)
% 
% K.H.Kraemer, Mar 2020

Y = varargin{1};
x = varargin{2};
tau = varargin{3};

if size(Y,1)<size(Y,2)
    Y = Y';
end

if size(x,1)<size(x,2)
    x = x';
end

N = size(Y,1); 

timespan_diff = tau;
M = N - timespan_diff;

Y2 = zeros(M,size(Y,2)+1);
Y2(:,1:size(Y,2)) = Y(1:M,:);

Y2(:,size(Y,2)+1) = x(1+tau:N);

end

function [distances,comp_dist] = all_distances(varargin)
% all_distances2 computes all componentwise distances from one point 
% (a vector) to all other given points, but not all pairwise distances 
% between all points.
% This function is meant to determine the neighbourhood of a certain point
% without computing the whole distances matrix (as being done by the 
% pdist()-function)
%
% [distances, comp_dist] = all_distances(fid_point,Y,norm) computes all 
% distances, based from the input vector 'fid_point' to all other 
% points/vectors stored in the input 'Y' and stores it in output 
% 'distances'. The componentwise distances are stored in output 'comp_dist'
%
%
% K.H.Kraemer, Mar 2020

fid_point = varargin{1};
Y = varargin{2};


methLib={'euc','max'}; % the possible norms
try
    meth = varargin{3};
    if ~isa(meth,'char') || ~ismember(meth,methLib)
       warning(['Specified norm should be one of the following possible values:',...
           10,sprintf('''%s'' ',methLib{:})])
    end
catch
    meth = 'euc';
end


% compute distances to all other points
% YY = zeros(size(Y));
% for p = 1:size(YY,1)
%     YY(p,:) = fid_point;
% end
YY = repmat(fid_point,size(Y,1),1);
if strcmp(meth,'euc')
    distances = sqrt(sum((YY - Y) .^ 2, 2));
elseif strcmp(meth,'max')
    distances = max(abs(YY - Y),[],2);
end

if nargout > 1
    comp_dist = abs(YY - Y);
end

end